import os
import zipfile
import tempfile
import numpy as np

# Try to import pydicom and biopython
try:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import UID
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    
try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

from src.data.loader import load_dicom_zip, load_fasta_zip

def create_dummy_fasta_zip(zip_path):
    temp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(temp_dir, 'benign'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'malignant'), exist_ok=True)
    
    import random
    random.seed(42)
    # Create 20 benign FASTA
    for i in range(20):
        # Add random bases to make it noisy
        noise = "".join(random.choices(["A", "C", "G", "T"], k=random.randint(0, 10)))
        rec = SeqRecord(Seq("ATGCGTACGTAGCTAGCTAGC" * (i % 3 + 1) + noise), id=f"seq_{i}", description="")
        SeqIO.write(rec, os.path.join(temp_dir, 'benign', f'seq_{i}.fasta'), "fasta")
        
    # Create 20 malignant FASTA
    for i in range(20):
        noise = "".join(random.choices(["A", "C", "G", "T"], k=random.randint(0, 15)))
        # Occasional overlap with benign
        if random.random() < 0.2:
            rec = SeqRecord(Seq("ATGCGTACGTAGCTAGCTAGC" * (i % 3 + 1) + noise), id=f"seq_{i}", description="")
        else:
            rec = SeqRecord(Seq("ATGCGTACGTAGCTAGCTAGCCC" * (i % 3 + 1) + noise), id=f"seq_{i}", description="")
        SeqIO.write(rec, os.path.join(temp_dir, 'malignant', f'seq_{i}.fasta'), "fasta")
    
    with zipfile.ZipFile(zip_path, 'w') as z:
        for root, _, files in os.walk(temp_dir):
            for f in files:
                full_path = os.path.join(root, f)
                arcname = os.path.relpath(full_path, temp_dir)
                z.write(full_path, arcname=arcname)
        
    print(f"Created {zip_path}")

def create_dummy_dicom_zip(zip_path):
    temp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(temp_dir, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'tumor'), exist_ok=True)
    
    def create_dicom(path, pixel_value):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')
        file_meta.MediaStorageSOPInstanceUID = UID("1.2.3")
        file_meta.ImplementationClassUID = UID("1.2.3.4")
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        ds = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    import numpy as np
    import random
    from pydicom.dataset import Dataset, FileDataset
    import pydicom.uid
    import datetime
    
    np.random.seed(42)
    random.seed(42)
    
    def create_dicom(path, base_val, is_tumor=False):
        # Create minimal meta data
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        ds = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.PatientName = "Test^Patient"
        ds.PatientID = "123456"
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        
        ds.Rows = 64
        ds.Columns = 64
        ds.PixelRepresentation = 0
        ds.HighBit = 15
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        
        # Add noise
        noise = np.random.normal(0, 20, (64, 64)).astype(np.int16)
        
        if is_tumor and random.random() > 0.2:
            # Tumor: higher base intensity and distinct center blob
            pixel_array = np.full((64, 64), base_val, dtype=np.int16)
            pixel_array[20:40, 20:40] += 50
        else:
            # Normal: lower base intensity, occasionally resembling tumor due to noise
            pixel_array = np.full((64, 64), base_val - (20 if not is_tumor else 0), dtype=np.int16)
            
        pixel_array = np.clip(pixel_array + noise, 0, 1000).astype(np.uint16)
        ds.PixelData = pixel_array.tobytes()
        ds.save_as(path)
        
    for i in range(20):
        create_dicom(os.path.join(temp_dir, 'normal', f'img_{i}.dcm'), 100 + random.randint(-10, 10), is_tumor=False)
        create_dicom(os.path.join(temp_dir, 'tumor', f'img_{i}.dcm'), 130 + random.randint(-10, 10), is_tumor=True)
    
    with zipfile.ZipFile(zip_path, 'w') as z:
        for root, _, files in os.walk(temp_dir):
            for f in files:
                full_path = os.path.join(root, f)
                arcname = os.path.relpath(full_path, temp_dir)
                z.write(full_path, arcname=arcname)
        
    print(f"Created {zip_path}")

if __name__ == "__main__":
    if HAS_BIO:
        create_dummy_fasta_zip("dummy_fasta.zip")
        X, y, f, t = load_fasta_zip("dummy_fasta.zip")
        print("FASTA Loader Output:")
        print("X:", X.shape, "y:", y)
        print("Targets:", t)
        print("Features:", f)
    else:
        print("Skipping FASTA test (biopython not installed)")
        
    if HAS_PYDICOM:
        create_dummy_dicom_zip("dummy_dicom.zip")
        X, y, f, t = load_dicom_zip("dummy_dicom.zip")
        print("\nDICOM Loader Output:")
        print("X:", X.shape, "y:", y)
        print("Targets:", t)
        print("Features:", f)
    else:
        print("Skipping DICOM test (pydicom not installed)")
