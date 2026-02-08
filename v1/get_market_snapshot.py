import sys

def get_last_lines(file_path, num_lines=30):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            return "".join(lines[-num_lines:])
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    path = "projects/singularity/dataset.csv"
    print(get_last_lines(path))
