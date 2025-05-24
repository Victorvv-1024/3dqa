import argparse
import re
import subprocess
import sys

# NOTE: Global 'import torch' was removed to prevent script crashing before dependencies are fixed.

def run_subprocess(command):
    try:
        command = [str(c) for c in command] # Ensure all command parts are strings
        print(f"Running command: {' '.join(command)}")
        process = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True, # Use text=True for universal_newlines
                                   bufsize=1) # Line buffering

        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line.strip())
            process.stdout.close()
        
        if process.stderr:
            for line in iter(process.stderr.readline, ''):
                # Print errors to stderr for better visibility if needed
                sys.stderr.write(f"STDERR: {line.strip()}\n")
            process.stderr.close()

        process.wait()
        return_code = process.returncode
        if return_code != 0:
            print(f'Command failed with return code {return_code} for: {" ".join(command)}')
        return return_code
    except Exception as e:
        print(f"An error occurred executing command: {' '.join(command)}")
        print(f"Exception: {e}")
        return 1 # Indicate failure

def pytorch3d_links():
    import torch # Import torch locally when needed
    cuda_version = torch.version.cuda
    pyt_torch_version = torch.__version__.split('+')[0] # e.g., "2.2.0"

    if cuda_version is None:
        print('Pytorch is cpu only. PyTorch3D installation will likely fail or be CPU only.')
        # Fallback or error, as this project likely needs CUDA
        # return "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cpu_pyt2.2.0/download.html" # Example CPU link
        raise NotImplementedError("PyTorch3D usually requires CUDA for this project.")

    cuda_version_str = cuda_version.replace('.', '') # e.g., "118"
    
    # --- YOU MUST VERIFY THIS pyt_version_str FOR PYTORCH3D ---
    # Option 1: If PyTorch3D URL uses '2.2.0' style for PyTorch version
    pyt_version_str = pyt_torch_version 
    # Option 2: If PyTorch3D URL uses '2.2' style
    # _pyt_parts = pyt_torch_version.split('.')
    # pyt_version_str = f"{_pyt_parts[0]}.{_pyt_parts[1]}"
    # Option 3: If PyTorch3D URL uses '220' style (less likely for recent versions)
    # pyt_version_str = pyt_torch_version.replace('.', '') 
    # --- END VERIFICATION ---
    # Example: if correct is 'pyt2.2.0', ensure pyt_version_str becomes '2.2.0'

    version_str = f'py3{sys.version_info.minor}_cu{cuda_version_str}_pyt{pyt_version_str}'
    links = f'https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html'
    print(f"Generated PyTorch3D find-links URL: {links}")
    return links

def mmcv_links():
    import torch # Import torch locally when needed
    cuda_version = torch.version.cuda
    if cuda_version is None:
        print('Pytorch is cpu only. MMCV-full installation will likely fail or be CPU only.')
    
    cuda_version_str = torch.version.cuda.replace('.', '')
    pyt_version_parts = torch.__version__.split('+')[0].split('.')
    pyt_version_mmcv = f"{pyt_version_parts[0]}.{pyt_version_parts[1]}" # e.g., "2.2" for PyTorch 2.2.0
    links = f'https://download.openmmlab.com/mmcv/dist/cu{cuda_version_str}/torch{pyt_version_mmcv}/index.html'
    print(f"Generated MMCV find-links URL: {links}")
    return links

def install_package(line_from_req_file):
    processed_line = line_from_req_file.split('#')[0].strip()
    if not processed_line:
        print(f"Skipping empty or comment-only line: {line_from_req_file}")
        return 0 # Success for an empty/comment line

    pat = '(' + '|'.join(['>=', '==', '>', '<', '<=', '@']) + ')'
    parts = re.split(pat, processed_line, maxsplit=1)
    package_name = parts[0].strip()
    
    print(f'Attempting to install: {package_name} (from processed line: "{processed_line}")')

    if package_name == 'pytorch3d':
        links = pytorch3d_links()
        # Pass the package name 'pytorch3d' and if version is pinned e.g. 'pytorch3d==0.7.0'
        # use processed_line. If only 'pytorch3d' then just package_name.
        # For now, assuming requirements just list 'pytorch3d' without version.
        # If your requirements/run.txt pins a version like 'pytorch3d==X.Y.Z', 
        # then use 'processed_line' instead of "'pytorch3d'" in the command.
        return run_subprocess(
            [sys.executable, '-m', 'pip', 'install', 'pytorch3d', '--no-cache-dir', '-f', links])
    elif package_name == 'mmcv' or package_name == 'mmcv-full':
        links = mmcv_links()
        return run_subprocess( # 'processed_line' will be e.g. "mmcv-full==2.2.0"
            [sys.executable, '-m', 'pip', 'install', processed_line, '--no-cache-dir', '-f', links])
    elif package_name == 'MinkowskiEngine':
        ret_code = run_subprocess([sys.executable, '-m', 'pip', 'install', 'ninja', '--no-cache-dir'])
        if ret_code != 0: return ret_code
        return run_subprocess([
            sys.executable, '-m', 'pip', 'install', '-U',
            'git+https://github.com/NVIDIA/MinkowskiEngine', '--no-deps', '--no-cache-dir'
        ])
    else:
        return run_subprocess([sys.executable, '-m', 'pip', 'install', processed_line, '--no-cache-dir'])

def install_requires(fname):
    print(f"\nProcessing requirements from: {fname}")
    with open(fname, 'r') as f:
        for line_content in f.readlines():
            ret_code = install_package(line_content)
            if ret_code != 0 and ret_code is not None: # Check if install_package indicated an error
                print(f"Stopping script due to error installing from line: {line_content.strip()}")
                sys.exit(ret_code) # Exit script if a package installation fails

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Install EmbodiedQA from pre-built package.')
    parser.add_argument('mode', nargs='?', default='all',
                        help="Installation mode: 'base', 'visual', 'run', or 'all'. Defaults to 'all'.")
    args = parser.parse_args()

    print(f"Starting installation in mode: {args.mode}")
    
    # The manual installations of numpy, transformers, torch-scatter should have been done before this.
    
    install_requires('requirements/base.txt') # [File 34]
    if args.mode == 'visual' or args.mode == 'all':
        install_requires('requirements/visual.txt') # [File 32]
    if args.mode == 'run' or args.mode == 'all':
        install_requires('requirements/run.txt') # [File 33]
    
    print("\nAttempting editable install of current project...")
    run_subprocess([sys.executable, '-m', 'pip', 'install', '-e', '.', '--no-cache-dir'])
    print("\nInstallation script finished.")

# REMOVE the hardcoded mmcv-full install block that was here.