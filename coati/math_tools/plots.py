from io import BytesIO
import base64

def get_smiles_image(s, size=(300, 300)):
    """
    Generate a PIL image from a smiles string.
    """
    from rdkit import Chem
    import rdkit.Chem.Draw

    return rdkit.Chem.Draw.MolToImage(Chem.MolFromSmiles(s), size=size)

def wrapped_get_smiles_image(X, size=(300, 300)):
    if not type(X) == str or X is None:
        return get_smiles_image('C', size=size)
    try:
        return get_smiles_image(X, size=size)
    except Exception as Ex:
        return get_smiles_image('C', size=size)
    
def image_formatter2(im, size=(90, 90)):
    with BytesIO() as buffer:
        im.thumbnail(size)
        im.save(buffer, 'png')
        data = base64.encodebytes(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{data}"  # <--------- this prefix is crucial
