# Description:
#   TensorFlow Lite TSS example.

load("//tensorflow/lite:build_def.bzl", "tflite_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "model_lib",
    srcs = [
        "models/tss_model.cc",
    ],
    hdrs = [
        "models/tss_model.h",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "tss",
    srcs = [
        "tss.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":model_lib",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
)
