# General Purpose Imports
from pathlib import Path

# Archive Imports
import zipfile
import py7zr
import multivolumefile


class DatasetExtractor:

    @staticmethod
    def extract_zip_archive(zip_file_path:Path, target_dir:Path, dirs_to_be_extracted = list()):
        """Extract specified contents from zip archive, extract all contents if not specified"""

        with zipfile.ZipFile(zip_file_path, 'r') as zip:
            #If dirs_to_be_extracted is an empty list, extract entire archive and exit
            if not dirs_to_be_extracted:
                zip.extractall(target_dir); return
            #Otherwise, extract all files under specified dirs
            #For each file in archive, extract if it's under any specified dir
            for member in zip.infolist():
                for foldername in dirs_to_be_extracted:
                    if foldername in member.filename:
                        #TODO: Add tqdm progress bar for extraction
                        zip.extract(member, target_dir)

    @staticmethod    
    def extract_multivolume_archive(multivolume_file_path:Path, target_dir:Path) -> None:
        """Extract all contents of a multivolume 7zip archive""" 

        with multivolumefile.open(multivolume_file_path, mode = 'rb') as multi_archive:
            with py7zr.SevenZipFile(multi_archive, 'r') as archive: # type: ignore
                archive.extractall(path = target_dir)

    @staticmethod
    def validate_extraction(val_dir:Path, val_files: list):
        """Check if val_dir contains all files listed under val_files, return list of missing files"""
        pass