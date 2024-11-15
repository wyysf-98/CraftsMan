## Build Docker Environment and use with GPU Support

Before you can use this Docker environment, you need to have the following:

- Docker installed on your system
- NVIDIA drivers installed on your system
- NVIDIA Container Toolkit installed on your system


### Build and Run
1. Build docker image:
   ```sh
   docker build -t CraftsMan:latest .
   ```
2. Start the docker container:
   ```sh
   docker run --gpus all -it CraftsMan:latest /bin/bash
   ```
3. Clone the repository:
   ```sh
   git clone git@github.com:wyysf-98/CraftsMan.git
   ```

## Troubleshooting

If you encounter any issues with the Docker environment with GPU support, please check the following:

- Make sure that you have installed the NVIDIA drivers and NVIDIA Container Toolkit on your system.
- Make sure that you have specified the --gpus all option when starting the Docker container.
- Make sure that your deep learning application is configured to use the GPU.