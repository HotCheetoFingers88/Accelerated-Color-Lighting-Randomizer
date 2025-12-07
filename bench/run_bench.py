# Small runner to build CSV comparing CPU vs GPU timings by invoking the compiled binary.
import subprocess, csv, sys

input_image = sys.argv[1] if len(sys.argv) > 1 else 'input.png'
num_copies = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 8

# build binary assumed in build/
proc = subprocess.run(['./randomizer', input_image, str(num_copies), str(batch_size)], capture_output=True, text=True)
print(proc.stdout)
with open('bench_results.txt', 'w') as f:
    f.write(proc.stdout)

