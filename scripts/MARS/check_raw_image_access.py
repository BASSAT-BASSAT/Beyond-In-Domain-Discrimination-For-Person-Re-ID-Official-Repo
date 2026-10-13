import io

import deeplake
import numpy as np
from PIL import Image


ds = deeplake.load(
    "hub://activeloop/mars-train",
    read_only=True,
)


print("=" * 90)
print("MARS DEEPLAKE RAW IMAGE ACCESS CHECK")
print("=" * 90)


sample = ds["images"][0]


print("\nSample type:")
print(type(sample))


print("\nImage tensor:")
print("htype:", ds["images"].htype)

try:
    print(
        "sample compression:",
        ds["images"].meta.sample_compression
    )
except Exception as e:
    print(
        "sample compression unavailable:",
        repr(e)
    )


print("\nSample info:")

try:
    print(sample.sample_info)
except Exception as e:
    print(
        "sample_info unavailable:",
        repr(e)
    )


# ============================================================
# Reconstructed official filename
# ============================================================

def get_text(key):
    data = ds[key][0].data()

    text = data.get("text")

    if isinstance(text, list):
        return str(text[0])

    return str(text)


pid = get_text(
    "pedestrian_ids"
)

camera = get_text(
    "camera_nos"
)

track = get_text(
    "track_nos"
)

frame = get_text(
    "frame_nos"
)


filename = (
    pid
    + camera
    + track
    + frame
    + ".jpg"
)


print("\nReconstructed filename:")
print(filename)


# ============================================================
# Normal decoded array
# ============================================================

arr = sample.numpy()

print("\nDecoded numpy:")
print("shape:", arr.shape)
print("dtype:", arr.dtype)
print(
    "raw decoded byte length:",
    len(arr.tobytes())
)


# ============================================================
# Check sample.tobytes()
# ============================================================

print("\nTesting sample.tobytes():")


if not hasattr(
    sample,
    "tobytes"
):
    print(
        "NO tobytes() method"
    )

else:

    try:

        raw = sample.tobytes()

        print(
            "tobytes length:",
            len(raw)
        )

        print(
            "first 16 bytes:",
            raw[:16].hex()
        )

        print(
            "JPEG magic:",
            raw[:3].hex()
            in {
                "ffd8ff",
            }
        )


        try:

            im = Image.open(
                io.BytesIO(
                    raw
                )
            )

            im.load()

            print(
                "PIL can open tobytes(): YES"
            )

            print(
                "format:",
                im.format
            )

            print(
                "size:",
                im.size
            )

            print(
                "mode:",
                im.mode
            )


            decoded_raw = np.asarray(
                im.convert(
                    "RGB"
                )
            )


            decoded_ds = arr

            if (
                decoded_ds.ndim == 2
            ):
                decoded_ds = np.stack(
                    [
                        decoded_ds,
                        decoded_ds,
                        decoded_ds,
                    ],
                    axis=-1,
                )

            elif (
                decoded_ds.ndim == 3
                and
                decoded_ds.shape[-1] == 1
            ):
                decoded_ds = np.repeat(
                    decoded_ds,
                    3,
                    axis=-1,
                )


            if (
                decoded_ds.ndim == 3
                and
                decoded_ds.shape[-1] >= 3
            ):
                decoded_ds = (
                    decoded_ds[
                        ...,
                        :3
                    ]
                )


            print(
                "Decoded pixels identical:",
                np.array_equal(
                    decoded_raw,
                    decoded_ds
                )
            )


        except Exception as e:

            print(
                "PIL can open tobytes(): NO"
            )

            print(
                "Reason:",
                repr(e)
            )


    except Exception as e:

        print(
            "tobytes() failed:"
        )

        print(
            repr(e)
        )


print("\nDONE")
