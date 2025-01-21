import os
from openfold.data.data_pipeline import DataPipeline
from openfold.data import parsers, templates
import torch
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for more detail
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)

def process_msa_directory(base_dir: str, output_dir: str):
    """Process all MSAs in the directory structure and save features"""
    # Initialize pipeline
    data_pipeline = DataPipeline(template_featurizer=None)
    
    # Get all protein directories
    protein_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    logger.info(f"Found {len(protein_dirs)} protein directories")

    for protein_id in protein_dirs:
        try:
            protein_dir = os.path.join(base_dir, protein_id)
            output_path = os.path.join(output_dir, f"{protein_id}_msa_features.pt")
            
            # Skip if already processed
            if os.path.exists(output_path):
                logger.info(f"Skipping {protein_id} - already processed")
                continue
                
            logger.info(f"Processing {protein_id}")
            
            # Construct path to a3m file
            a3m_dir = os.path.join(protein_dir, 'a3m')
            if not os.path.exists(a3m_dir):
                logger.warning(f"No a3m directory found for {protein_id}")
                continue
                
            # Find the .a3m file
            a3m_files = [f for f in os.listdir(a3m_dir) if f.endswith('.a3m')]
            if not a3m_files:
                logger.warning(f"No .a3m files found for {protein_id}")
                continue
                
            a3m_path = os.path.join(a3m_dir, a3m_files[0])
            logger.info(f"Found A3M file: {a3m_path}")
            
            # Read and parse the a3m file directly
            try:
                with open(a3m_path, "r") as fp:
                    msa, deletion_matrix = parsers.parse_a3m(fp.read())
                
                msa_data = {
                    "msa": msa,
                    "deletion_matrix": deletion_matrix
                }
                
                logger.info(f"Successfully parsed MSA for {protein_id}")
                logger.info(f"Number of sequences: {len(msa)}")
                logger.info(f"Sequence length: {len(msa[0]) if msa else 0}")
                
                # Create output directory if needed
                os.makedirs(output_dir, exist_ok=True)
                
                # Save features
                torch.save(msa_data, output_path)
                logger.info(f"Saved MSA data for {protein_id}")
                
            except Exception as e:
                logger.error(f"Error parsing A3M file for {protein_id}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            
        except Exception as e:
            logger.error(f"Error processing {protein_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--msa_dir', 
        type=str, 
        default='/cbica/home/shahroha/projects/AF-DIT/atlas/MSA/',
        help='Directory containing MSA folders'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/cbica/home/shahroha/projects/AF-DIT/atlas/alignment_dir/',
        help='Where to save processed features'
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting MSA preprocessing")
    logger.info(f"MSA directory: {args.msa_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    process_msa_directory(args.msa_dir, args.output_dir)
    logger.info("Processing complete")