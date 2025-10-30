"""
Atomic properties for all elements.

Source: Standard atomic data tables
"""

# All available elements with their properties
ATOMIC_PROPERTIES = {
    # Refractory metals (your main elements)
    'Mo': {
        'atomic_radius': 1.47,      # Angstrom
        'atomic_mass': 95.95,       # amu
        'electronegativity': 2.16,  # Pauling scale
        'valence': 6
    },
    'Nb': {
        'atomic_radius': 1.64,
        'atomic_mass': 92.91,
        'electronegativity': 1.6,
        'valence': 5
    },
    'Ta': {
        'atomic_radius': 1.46,
        'atomic_mass': 180.95,
        'electronegativity': 1.5,
        'valence': 5
    },
    'W': {
        'atomic_radius': 1.47,
        'atomic_mass': 183.84,
        'electronegativity': 2.36,
        'valence': 6
    },
    
    # Additional elements (extend as needed)
    'O': {
        'atomic_radius': 0.66,
        'atomic_mass': 15.999,
        'electronegativity': 3.44,
        'valence': 2
    },
    'Fe': {
        'atomic_radius': 1.56,
        'atomic_mass': 55.845,
        'electronegativity': 1.83,
        'valence': 3
    },
    'Cr': {
        'atomic_radius': 1.66,
        'atomic_mass': 51.996,
        'electronegativity': 1.66,
        'valence': 3
    },
    'Ni': {
        'atomic_radius': 1.49,
        'atomic_mass': 58.693,
        'electronegativity': 1.91,
        'valence': 2
    },
    'V': {
        'atomic_radius': 1.71,
        'atomic_mass': 50.942,
        'electronegativity': 1.63,
        'valence': 5
    },
    'Ti': {
        'atomic_radius': 1.76,
        'atomic_mass': 47.867,
        'electronegativity': 1.54,
        'valence': 4
    },
    'Zr': {
        'atomic_radius': 1.75,
        'atomic_mass': 91.224,
        'electronegativity': 1.33,
        'valence': 4
    },
    'Hf': {
        'atomic_radius': 1.75,
        'atomic_mass': 178.49,
        'electronegativity': 1.3,
        'valence': 4
    },
    'Re': {
        'atomic_radius': 1.51,
        'atomic_mass': 186.207,
        'electronegativity': 1.9,
        'valence': 7
    },
    'Os': {
        'atomic_radius': 1.44,
        'atomic_mass': 190.23,
        'electronegativity': 2.2,
        'valence': 4
    },
    'Ir': {
        'atomic_radius': 1.41,
        'atomic_mass': 192.217,
        'electronegativity': 2.20,
        'valence': 4
    },
    'Pt': {
        'atomic_radius': 1.39,
        'atomic_mass': 195.084,
        'electronegativity': 2.28,
        'valence': 4
    },
    'Al': {
        'atomic_radius': 1.82,
        'atomic_mass': 26.982,
        'electronegativity': 1.61,
        'valence': 3
    },
    'Cu': {
        'atomic_radius': 1.45,
        'atomic_mass': 63.546,
        'electronegativity': 1.90,
        'valence': 2
    },
    'Mn': {
        'atomic_radius': 1.61,
        'atomic_mass': 54.938,
        'electronegativity': 1.55,
        'valence': 2
    },
    'Co': {
        'atomic_radius': 1.52,
        'atomic_mass': 58.933,
        'electronegativity': 1.88,
        'valence': 2
    }
}

# Property keys (for consistent ordering)
PROPERTY_KEYS = ['atomic_radius', 'atomic_mass', 'electronegativity', 'valence']


def get_element_properties(element: str) -> dict:
    """
    Get atomic properties for an element.
    
    Parameters:
    -----------
    element : str
        Element symbol (e.g., 'Mo', 'W')
    
    Returns:
    --------
    properties : dict
        Dictionary with atomic properties
    
    Raises:
    -------
    ValueError
        If element is not in database
    """
    if element not in ATOMIC_PROPERTIES:
        available = ', '.join(sorted(ATOMIC_PROPERTIES.keys()))
        raise ValueError(
            f"Unknown element: {element}\n"
            f"Available elements: {available}"
        )
    
    return ATOMIC_PROPERTIES[element]


def get_all_elements() -> list:
    """Get list of all available elements"""
    return sorted(ATOMIC_PROPERTIES.keys())


if __name__ == "__main__":
    print("="*70)
    print("ATOMIC PROPERTIES DATABASE")
    print("="*70)
    
    print(f"\nAvailable elements: {len(ATOMIC_PROPERTIES)}")
    print(f"  {', '.join(get_all_elements())}")
    
    print("\nExample properties:")
    for elem in ['Mo', 'W', 'O', 'Fe']:
        if elem in ATOMIC_PROPERTIES:
            props = ATOMIC_PROPERTIES[elem]
            print(f"\n  {elem}:")
            for key in PROPERTY_KEYS:
                print(f"    {key}: {props[key]}")
    
    print("\n" + "="*70)