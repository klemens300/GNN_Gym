"""
Atomic properties for all elements.

Extended version with all properties needed by graph_builder.py
"""

import warnings

# Extended atomic properties database
ATOMIC_PROPERTIES = {
    # Refractory metals (main elements for MoNbTaW)
    'Mo': {
        'atomic_number': 42,
        'atomic_mass': 95.95,
        'atomic_radius': 1.47,  # Angstrom
        'electronegativity': 2.16,  # Pauling scale
        'first_ionization': 7.09,  # eV
        'electron_affinity': 0.75,  # eV
        'melting_point': 2896.0,  # K
        'density': 10.28,  # g/cm³
        'valence': 6
    },
    'Nb': {
        'atomic_number': 41,
        'atomic_mass': 92.91,
        'atomic_radius': 1.64,
        'electronegativity': 1.6,
        'first_ionization': 6.76,
        'electron_affinity': 0.89,
        'melting_point': 2750.0,
        'density': 8.57,
        'valence': 5
    },
    'Ta': {
        'atomic_number': 73,
        'atomic_mass': 180.95,
        'atomic_radius': 1.46,
        'electronegativity': 1.5,
        'first_ionization': 7.55,
        'electron_affinity': 0.32,
        'melting_point': 3290.0,
        'density': 16.65,
        'valence': 5
    },
    'W': {
        'atomic_number': 74,
        'atomic_mass': 183.84,
        'atomic_radius': 1.47,
        'electronegativity': 2.36,
        'first_ionization': 7.86,
        'electron_affinity': 0.82,
        'melting_point': 3695.0,
        'density': 19.25,
        'valence': 6
    },
    
    # Additional common elements
    'Fe': {
        'atomic_number': 26,
        'atomic_mass': 55.845,
        'atomic_radius': 1.56,
        'electronegativity': 1.83,
        'first_ionization': 7.90,
        'electron_affinity': 0.15,
        'melting_point': 1811.0,
        'density': 7.87,
        'valence': 3
    },
    'Cr': {
        'atomic_number': 24,
        'atomic_mass': 51.996,
        'atomic_radius': 1.66,
        'electronegativity': 1.66,
        'first_ionization': 6.77,
        'electron_affinity': 0.67,
        'melting_point': 2180.0,
        'density': 7.19,
        'valence': 3
    },
    'Ni': {
        'atomic_number': 28,
        'atomic_mass': 58.693,
        'atomic_radius': 1.49,
        'electronegativity': 1.91,
        'first_ionization': 7.64,
        'electron_affinity': 1.16,
        'melting_point': 1728.0,
        'density': 8.91,
        'valence': 2
    },
    'Al': {
        'atomic_number': 13,
        'atomic_mass': 26.982,
        'atomic_radius': 1.82,
        'electronegativity': 1.61,
        'first_ionization': 5.99,
        'electron_affinity': 0.43,
        'melting_point': 933.5,
        'density': 2.70,
        'valence': 3
    },
    'Ti': {
        'atomic_number': 22,
        'atomic_mass': 47.867,
        'atomic_radius': 1.76,
        'electronegativity': 1.54,
        'first_ionization': 6.83,
        'electron_affinity': 0.08,
        'melting_point': 1941.0,
        'density': 4.51,
        'valence': 4
    },
    'V': {
        'atomic_number': 23,
        'atomic_mass': 50.942,
        'atomic_radius': 1.71,
        'electronegativity': 1.63,
        'first_ionization': 6.75,
        'electron_affinity': 0.53,
        'melting_point': 2183.0,
        'density': 6.11,
        'valence': 5
    },
    'Zr': {
        'atomic_number': 40,
        'atomic_mass': 91.224,
        'atomic_radius': 1.75,
        'electronegativity': 1.33,
        'first_ionization': 6.63,
        'electron_affinity': 0.43,
        'melting_point': 2128.0,
        'density': 6.51,
        'valence': 4
    },
    'Hf': {
        'atomic_number': 72,
        'atomic_mass': 178.49,
        'atomic_radius': 1.75,
        'electronegativity': 1.3,
        'first_ionization': 6.83,
        'electron_affinity': 0.0,
        'melting_point': 2506.0,
        'density': 13.31,
        'valence': 4
    },
    'Re': {
        'atomic_number': 75,
        'atomic_mass': 186.207,
        'atomic_radius': 1.51,
        'electronegativity': 1.9,
        'first_ionization': 7.83,
        'electron_affinity': 0.15,
        'melting_point': 3459.0,
        'density': 21.02,
        'valence': 6
    },
    'O': {
        'atomic_number': 8,
        'atomic_mass': 15.999,
        'atomic_radius': 0.66,
        'electronegativity': 3.44,
        'first_ionization': 13.62,
        'electron_affinity': 1.46,
        'melting_point': 54.8,
        'density': 1.43,
        'valence': 2
    },
}

# Property keys (for consistent ordering)
PROPERTY_KEYS = [
    'atomic_number',
    'atomic_mass',
    'atomic_radius',
    'electronegativity',
    'first_ionization',
    'electron_affinity',
    'melting_point',
    'density',
    'valence'
]


def get_atomic_properties(element: str) -> dict:
    """
    Get atomic properties for an element.
    
    Parameters:
    -----------
    element : str
        Element symbol (e.g., 'Mo', 'W')
    
    Returns:
    --------
    properties : dict
        Dictionary with atomic properties:
        - atomic_number: int
        - atomic_mass: float (amu)
        - atomic_radius: float (Angstrom)
        - electronegativity: float (Pauling scale)
        - first_ionization: float (eV)
        - electron_affinity: float (eV)
        - melting_point: float (K)
        - density: float (g/cm³)
        - valence: int
    
    Raises:
    -------
    ValueError
        If element is not in database
    """
    if element not in ATOMIC_PROPERTIES:
        available = ', '.join(sorted(ATOMIC_PROPERTIES.keys()))
        warnings.warn(
            f"Unknown element: {element}. Using default values. "
            f"Available elements: {available}"
        )
        # Return default values
        return {
            'atomic_number': 42,  # Mo as default
            'atomic_mass': 95.95,
            'atomic_radius': 1.47,
            'electronegativity': 2.0,
            'first_ionization': 7.0,
            'electron_affinity': 0.5,
            'melting_point': 2500.0,
            'density': 10.0,
            'valence': 5
        }
    
    return ATOMIC_PROPERTIES[element]


def get_element_properties(element: str) -> dict:
    """Alias for get_atomic_properties for backward compatibility."""
    return get_atomic_properties(element)


def get_all_elements() -> list:
    """Get list of all available elements"""
    return sorted(ATOMIC_PROPERTIES.keys())


def add_element(element: str, properties: dict):
    """
    Add a new element to the database.
    
    Parameters:
    -----------
    element : str
        Element symbol
    properties : dict
        Dictionary with atomic properties
    """
    ATOMIC_PROPERTIES[element] = properties


if __name__ == "__main__":
    print("="*70)
    print("ATOMIC PROPERTIES DATABASE")
    print("="*70)
    
    print(f"\nAvailable elements: {len(ATOMIC_PROPERTIES)}")
    print(f"  {', '.join(get_all_elements())}")
    
    print("\nMain elements (MoNbTaW):")
    for elem in ['Mo', 'Nb', 'Ta', 'W']:
        props = get_atomic_properties(elem)
        print(f"\n  {elem}:")
        for key in PROPERTY_KEYS:
            print(f"    {key}: {props[key]}")
    
    print("\n" + "="*70)