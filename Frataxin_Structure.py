# ===================[ Structural Bioinformatics Analysis of the Frataxin (FXN) Protein Involved in Friedreich's Ataxia ]======================


# --------------------------------------------------------- Import Libraries --------------------------------------------------------------

from Bio import PDB
from Bio.PDB.internal_coords import *
from collections import defaultdict
from Bio import ExPASy, SwissProt
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------- Streamlit Set Up Info -------------------------------------------------------
st.title(":grey[Structural Bioinformatics Analysis of the Frataxin (FXN) Protein Involved in Friedreich's Ataxia]")

# -------------------------------------Extract information about protein and related gene from SwissProt records----------------------------

query_id = "Q16595"
handle = ExPASy.get_sprot_raw(query_id)
record = SwissProt.read(handle)

# Print relevant information 
col1, col2, col3 = st.columns(3)

with col1: 
    st.header(":violet[Organism] ")
    st.divider()
    st.write(f":violet[Species:] :grey[{record.organism}]")
    st.write(f":violet[Classification:] :grey[{', '.join(record.organism_classification)}]")
    
with col2:
    st.header(":violet[Gene ]")
    st.divider()
    st.write(f":violet[Gene Name:] :grey[{record.gene_name[0]['Name']}]")
    st.write(f":violet[Synonyms:] :grey[{', '.join(record.gene_name[0]['Synonyms'])}]")

with st.expander(":violet[Description]"):
    st.write(f"{', '.join(record.comments).replace('., ', '\n\n')}")
st.divider()

with st.expander(":violet[Related Literature]"):
    for ref in record.references:
        st.write(f":violet[Authors:] :grey[{ref.authors}]")
        st.write(f":violet[Title:] :grey[{ref.title}]")
        st.divider()

st.divider()

# --------------------------------------- Retrieve frataxin structure from PDB ------------------------------------------------------

# Frataxin PDB code  
pdb_code = '1EKG'

# Initialize the PDBList and PDBP objects from Biopython module
pdbl = PDB.PDBList() # Allows download of PDB files
parser = PDB.PDBParser() # Reads & Parses PDB files (atoms, residues, chains, etc.)

# Download PDB file for Frataxin structure to create PDB structure object
pdbl.retrieve_pdb_file(pdb_code, file_format = 'pdb', pdir = '.')

# Create Structure object from PDB file
structure = parser.get_structure(pdb_code, f"pdb{pdb_code}.ent" )

# Basic structure analysis
# The overall layout of a Structure object follows the so-called SMCRA (Structure/Model/Chain/Residue/Atom) architecture:
# • A structure consists of models
# • A model consists of chains
# • A chain consists of residues
# • A residue consists of atoms


left, right = st.columns(2)

# Function to print the header information of a PDB structure
def print_pdb_headers(headers, indent=0):
    ind_text = ' ' * indent
    with left:
        st.header(":violet[PDB Record]")
        st.divider()
        for header, content in headers.items():
                st.write(f":grey[{ind_text:<20} :violet[{header}]: {content}]")

# Print the header information for the PDB structure
print_pdb_headers(structure.header)

# Initialize empty dictionary 
data = []

# Structure Analysis 
models = structure.get_models()
for model in models:
    for chain in model:
        with right:
            st.header(":violet[Protein Structure]")
            st.divider()
            st.write(f":violet[{model}]")
            st.write(f":violet[Chain:] :grey[{chain.id}.]")
            st.write(f":violet[Number of residues:] :grey[{len(chain)}.]")
            st.write(f":violet[Number of atoms:] :grey[{len(list(chain.get_atoms()))}]")
        for residue in chain:
            residue_name = residue.get_resname()
            residue_id = str(residue.id)
            for atom in residue:
                atom_name = atom.get_name()
                data.append({"Residue": residue_name, "Residue ID": residue_id, "Atom": atom_name})

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame as a table
st.divider()
st.markdown("### :violet[Chain Residues & Atoms of Protein Structure]")
st.dataframe(df)

st.divider()

# --------------------------------------------------------- Basic Structure Stats -------------------------------------------------------
st.markdown("### :violet[Structure Stats]")

# Initialize defaultdicts to store statistics
atom_cnt = defaultdict(int)
atom_chain = defaultdict(int)
atom_res_types = defaultdict(int)

# Loop through all atoms in the structure to calculate statistics
for atom in structure.get_atoms():
    my_residue = atom.parent
    my_chain = my_residue.parent
    atom_chain[my_chain.id] += 1
    if my_residue.resname != 'HOH':
        atom_cnt[atom.element] += 1
    atom_res_types[my_residue.resname] += 1

# Print statistics of atom count per residue type, chain, and atom element
st.write(f":violet[Residue Types:] :grey[{dict(atom_res_types)}]")
st.write(f":violet[Number of atoms:] :grey[{dict(atom_chain)['A']}]")
st.write(f":violet[Atom Types:] :grey[{dict(atom_cnt)}]")

st.divider()

# ---------------------------------------------- Extract Polypeptides from Structure Object ----------------------------------------------

st.markdown("### :violet[Protein Structure Polypeptides]")

# Instance of polypeptide builder object of PDB.Polypeptide module
ppb = PDB.Polypeptide.PPBuilder()

# Get structure polypeptides
polypeptide_list = ppb.build_peptides(structure)

# Get sequence of the protein's polypeptides
for polypeptide in polypeptide_list:
    st.write(":violet[Polypeptide Residues:]")
    st.write(polypeptide)
    seq = polypeptide.get_sequence()
    st.write(f":violet[Polypeptide Sequence:] :grey[{seq}]")

st.write(f":violet[Sequence Length:] :grey[{len(seq)}]")
st.divider()

# ------------------------------------------------------------------- Mass ----------------------------------------------------------------

# Extract unique residue IDs from the PDB structure
my_residues = set()
for residue in structure.get_residues():
    my_residues.add(residue.id[0])

# Define a function 'get_mass' to calculate the mass of a group of atoms, with an optional filtering function
def get_mass(atoms, accept_fun=lambda atom: atom.parent.id[0] != 'W'):
    return sum([atom.mass for atom in atoms if accept_fun(atom)])

# Calculate the mass of atoms in each chain of the PDB structure and store the results in a DataFrame
chain_names = [chain.id for chain in structure.get_chains()]
my_mass = np.ndarray((len(chain_names), 3))
for i, chain in enumerate(structure.get_chains()):
    my_mass[i, 0] = get_mass(chain.get_atoms())
    my_mass[i, 1] = get_mass(chain.get_atoms(), accept_fun=lambda atom: atom.parent.id[0] not in [' ', 'W'])
    my_mass[i, 2] = get_mass(chain.get_atoms(), accept_fun=lambda atom: atom.parent.id[0] == 'W')  
masses = pd.DataFrame(my_mass, index=chain_names, columns=['No Water', 'Zincs', 'Water'])

st.markdown("#### :violet[Calculate mass of atoms in each chain of the protein's molecular structure]")
st.dataframe(masses)

# Define a function 'get_center' to calculate the center of a group of atoms, with an optional weighting function
def get_center(atoms, weight_fun=lambda atom: 1 if atom.parent.id[0] != 'W' else 0):
    xsum = ysum = zsum = 0.0
    acum = 0.0
    for atom in atoms:
        x, y, z = atom.coord
        weight = weight_fun(atom)
        acum += weight
        xsum += weight * x
        ysum += weight * y
        zsum += weight * z
    return xsum / acum, ysum / acum, zsum / acum

# Calculate the center of all atoms and atoms weighted by mass for each chain in the PDB structure
#st.write(get_center(structure.get_atoms()))
#st.write(get_center(structure.get_atoms(), weight_fun=lambda atom: atom.mass if atom.parent.id[0] != 'W' else 0))

my_center = np.ndarray((len(chain_names), 6))
for i, chain in enumerate(structure.get_chains()):
    x, y, z = get_center(chain.get_atoms())
    my_center[i, 0] = x
    my_center[i, 1] = y
    my_center[i, 2] = z
    x, y, z = get_center(chain.get_atoms(), weight_fun=lambda atom: atom.mass if atom.parent.id[0] != 'W' else 0)
    my_center[i, 3] = x
    my_center[i, 4] = y
    my_center[i, 5] = z

st.markdown("#### :violet[Calculate geometric and mass-weighted centers of chains in the protein's molecular structure]")
weights = pd.DataFrame(my_center, index=chain_names, columns=['X', 'Y', 'Z', 'X (Mass)', 'Y (Mass)', 'Z (Mass)'])
st.dataframe(weights)
