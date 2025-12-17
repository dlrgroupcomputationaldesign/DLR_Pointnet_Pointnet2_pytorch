from setuptools import setup, find_packages

def read_requirements(path="requirements.txt"):
    # utf-8-sig strips a UTF-8 BOM if present
    with open(path, encoding="utf-16") as f:
        reqs = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.append(line)
        return reqs
    
setup(
    name="inference",
    version="0.0.0",    #PEP 440
    description="Point cloud inference module.",
    packages=find_packages(),
    install_requires=read_requirements(),
)