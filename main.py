import os
import cv2
import argparse
import tempfile
import subprocess
import pyultrahdr
import numpy as np
from PIL import Image
from tqdm import tqdm


# SDR reference white level in nits.
SDR_WHITE = 203.0

# Target HDR peak brightness in nits.
PEAK_NITS = 1200.0

def sdr_to_hdr(input_file: str, output_file: str) -> None:

    # Open the input image and make sure it is represented as RGB.
    image = Image.open(input_file).convert("RGB")

    # Convert the Pillow image into a NumPy array containing 8-bit RGB values.
    sdr_uint8 = np.asarray(image, dtype=np.uint8)

    # SDR -> LINEAR
    # Normalize 8-bit RGB values from the range 0-255 to 0.0-1.0.
    sdr = sdr_uint8.astype(np.float32) / 255.0

    # Convert sRGB-encoded values into linear-light RGB values.
    # This uses the proper sRGB transfer function instead of a simple gamma 2.2 approximation.
    sdr_lin = np.where(
        sdr <= 0.04045,
        sdr / 12.92,
        np.power((sdr + 0.055) / 1.055, 2.4)
    ).astype(np.float32)

    # SDR -> HDR nits
    # Apply pyultrahdr's inverse tone mapping to expand the SDR linear-light image into HDR luminance values expressed in nits.
    hdr_nits = pyultrahdr.inverse_tone_map(
        sdr_lin,
        sdr_white=SDR_WHITE,
        peak_nits=PEAK_NITS,
        shadow_boost=1.3,
        hi_gamma=0.6,
    )

    # Keep the generated HDR values within the requested 0-1000 nit range.
    hdr_nits = np.clip(hdr_nits, 0.0, PEAK_NITS).astype(np.float32)

    # BT.709 -> BT.2020
    # Matrix used to convert the RGB color representation from BT.709 into the BT.2020 color space used by HDR10.
    bt709_to_bt2020 = np.array(
        [
            [0.6274040, 0.3292820, 0.0433136],
            [0.0690970, 0.9195400, 0.0113612],
            [0.0163916, 0.0880132, 0.8955950],
        ],
        dtype=np.float32
    )

    # Apply the BT.709 -> BT.2020 matrix to every pixel in the image.
    # The einsum operates over the height, width, and RGB channel dimensions.
    hdr_nits = np.einsum("ij,hwj->hwi", bt709_to_bt2020, hdr_nits, optimize=True)

    # Clip the converted HDR values again so they remain within the 0-1000 nit range.
    hdr_nits = np.clip(hdr_nits, 0.0, PEAK_NITS).astype(np.float32)

    # HDR nits -> PQ / ST.2084
    # PQ maps absolute luminance values up to 10,000 nits into normalized signal values from 0.0 to 1.0.
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 32.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 128.0
    c3 = 2392.0 / 128.0

    # Normalize absolute HDR luminance from 0-10,000 nits into the 0.0-1.0 range required by the PQ equation.
    L = np.clip(hdr_nits / 10000.0, 0.0, 1.0)

    # Apply the first exponent of the ST.2084 PQ transfer function.
    Lm1 = np.power(L, m1)

    # Apply the complete ST.2084 PQ transfer function.
    # The resulting values are normalized PQ code values from 0.0 to 1.0.
    pq = np.power((c1 + c2 * Lm1) / (1.0 + c3 * Lm1), m2)

    # PQ -> 16-bit
    # Convert normalized PQ values from 0.0-1.0 into 16-bit integer values from 0-65535, which allows the HDR frame to be stored as a 16-bit PNG.
    pq_uint16 = np.clip(np.round(pq * 65535.0), 0, 65535).astype(np.uint16)

    # SAVE 16-bit PQ HDR PNG
    # OpenCV expects color images in BGR channel order when writing.
    # Convert the RGB array to BGR before saving.
    # OpenCV supports 16-bit 3-channel PNG output.
    pq_bgr = cv2.cvtColor(pq_uint16, cv2.COLOR_RGB2BGR)

    # Write the 16-bit PQ HDR frame to the requested PNG path.
    cv2.imwrite(output_file, pq_bgr)


if __name__ == "__main__":
    # Command-line interface (CLI).

    parser = argparse.ArgumentParser(description="SDR to HDR converter.")
    parser.add_argument("-t", "--target", required=True, help="Source File")
    parser.add_argument("-d", "--dest", required=True, help="Destination directory")

    args = parser.parse_args()

    target_file = args.target
    destination = args.dest

    if not os.path.exists(target_file):
        print("Target file doesn't exists.")
        exit(1)
    
    if not os.path.isfile(target_file):
        print("Target file is not a file.")
        exit(1)

    sdr_dir = tempfile.TemporaryDirectory()
    hdr_dir = tempfile.TemporaryDirectory()

    # Extract frames from source video
    subprocess.run([
        "ffmpeg", "-hide_banner", "-y", "-v", "error", "-stats",
        "-i",
        target_file,
        "-map", "0:0",
        "-vsync", "0",
        "-c:v", "png",
        f"{sdr_dir.name}/%04d.png"
    ], check=True)


    files=sorted(f for f in os.listdir(sdr_dir.name) if f.endswith(".png"))

    # Convert sdr to hdr frames
    for file_name in tqdm(files, unit="frame"):
        sdr_to_hdr(
            input_file=os.path.join(sdr_dir.name, file_name),
            output_file=os.path.join(hdr_dir.name, os.path.splitext(file_name)[0] + ".png")
        )

    # Get source video's fps
    fps = subprocess.check_output(
        ["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=avg_frame_rate","-of","csv=p=0",target_file]
    ).decode().strip()

    # Encode frames into HDR10 video
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-y", "-v", "error", "-stats",
            "-framerate", fps,
            "-i", f"{hdr_dir.name}/%04d.png",
            "-c:v", "libx265",
            "-preset", "medium",
            "-pix_fmt", "yuv420p10le",
            "-x265-params",
            f"lossless=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50):max-cll={int(PEAK_NITS)},400",
            "-tag:v", "hvc1",
            "-color_primaries", "bt2020",
            "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc",
            "-movflags", "+faststart",
            destination
        ],
        check=True
    )

    sdr_dir.cleanup()
    hdr_dir.cleanup()
