# Amino acid reference table
# volume (rel. 0-1 scale), charge, and volume class as used in the volume classifier
#
# Volume class boundaries (Å³):
#   very_small: 60–90 | small: 108–117 | medium: 138–154 | large: 162–174 | very_large: 189–228
#
# Charge convention (Motone et al. 2024):
#   positive (+1) → decreases nanopore current
#   negative (-1) → increases nanopore current

# 1-letter: (full name,        charge,  rel. volume, volume class)
AA_REFERENCE = {
    'G': ('Glycine',        0,   0.00, 'very_small'),
    'A': ('Alanine',        0,   0.11, 'very_small'),
    'S': ('Serine',         0,   0.18, 'very_small'),
    'T': ('Threonine',      0,   0.26, 'small'),
    'C': ('Cysteine',       0,   0.20, 'small'),
    'P': ('Proline',        0,   0.28, 'small'),
    'D': ('Aspartate',     -1,   0.27, 'small'),
    'N': ('Asparagine',     0,   0.29, 'small'),
    'V': ('Valine',         0,   0.35, 'medium'),
    'H': ('Histidine',      0,   0.48, 'medium'),
    'E': ('Glutamate',     -1,   0.37, 'medium'),
    'Q': ('Glutamine',      0,   0.39, 'medium'),
    'I': ('Isoleucine',     0,   0.45, 'large'),
    'L': ('Leucine',        0,   0.45, 'large'),
    'M': ('Methionine',     0,   0.42, 'large'),
    'K': ('Lysine',        +1,   0.46, 'large'),
    'R': ('Arginine',      +1,   0.63, 'large'),
    'F': ('Phenylalanine',  0,   0.64, 'very_large'),
    'W': ('Tryptophan',     0,   0.79, 'very_large'),
    'Y': ('Tyrosine',       0,   0.67, 'very_large'),
}

if __name__ == '__main__':
    print(f"{'AA':<4} {'Name':<16} {'Charge':>7} {'Vol (rel)':>10} {'Volume Class'}")
    print('-' * 54)
    for aa, (name, charge, vol, cls) in AA_REFERENCE.items():
        charge_str = f'+{charge}' if charge > 0 else str(charge)
        print(f"{aa:<4} {name:<16} {charge_str:>7} {vol:>10.2f}   {cls}")
