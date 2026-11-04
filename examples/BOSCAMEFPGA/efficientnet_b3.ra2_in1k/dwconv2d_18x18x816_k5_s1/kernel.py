"""EfficientNet-B3 case bound to the staged W8A8 MMA convolution."""

# One public adapter invocation; the bound pipeline defines its raw stage grids.
KERNEL_CONFIG = {
    "family": "dwconv2d_f32",
    "implementation": "im2col_w8a8_depthwise",
    "kernel": "dwconv2d_im2col_w8a8",
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
        816,
        18,
        18
    ],
    "constexprs": {
        "H": 18,
        "W": 18,
        "C": 816,
        "KH": 5,
        "KW": 5,
        "SH": 1,
        "SW": 1,
        "PH": 2,
        "PW": 2,
        "OH": 18,
        "OW": 18
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
