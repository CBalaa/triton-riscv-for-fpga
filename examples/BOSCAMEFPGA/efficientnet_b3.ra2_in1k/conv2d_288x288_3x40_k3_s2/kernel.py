"""EfficientNet-B3 case bound to the staged W8A8 MMA convolution."""

# One public adapter invocation; the bound pipeline defines its raw stage grids.
KERNEL_CONFIG = {
    "family": "conv2d_f32",
    "implementation": "im2col_w8a8",
    "kernel": "conv2d_im2col_w8a8",
    "arguments": [
        {
            "name": "X",
            "rank": 4,
            "dtype": "f32"
        },
        {
            "name": "Weight",
            "rank": 4,
            "dtype": "f32"
        },
        {
            "name": "Bias",
            "rank": 1,
            "dtype": "f32"
        },
        {
            "name": "Out",
            "rank": 4,
            "dtype": "f32"
        }
    ],
    "shape": [
        1,
        3,
        288,
        288
    ],
    "constexprs": {
        "H": 288,
        "W": 288,
        "C": 3,
        "OC": 40,
        "KH": 3,
        "KW": 3,
        "SH": 2,
        "SW": 2,
        "PH": 1,
        "PW": 1,
        "OH": 144,
        "OW": 144
    },
    "grid": [
        1,
        1,
        1
    ],
    "roles": [],
    "metadata": {
        "dtype": "efficientnet-b3",
        "model": "timm/efficientnet_b3.ra2_in1k"
    },
    "kernel_module": "quantized_conv_kernels"
}
