# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

##############################################################################
# Docker commands (from project root directory)

# docker build \
#   --rm \
#   --tag gitlab-master.nvidia.com:5005/srl/scene_synthesizer/ubuntu1804_py37:latest \
#   .

# docker push gitlab-master.nvidia.com:5005/srl/scene_synthesizer/ubuntu1804_py37:latest

##############################################################################

# Set base image
FROM gitlab-master.nvidia.com:5005/srl/dockers/ubuntu1804_py37:latest

# Install openscad
RUN sudo add-apt-repository -y ppa:openscad/releases
RUN sudo apt-get update && sudo apt-get install -y openscad
