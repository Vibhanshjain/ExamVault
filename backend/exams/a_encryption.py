from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
import os
from django.conf import settings

def a_encryption(hash_id, key, t_id):
    message = [hash_id, key, t_id]

    # Generate RSA keys
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()

    # Save private key
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    prk_file = os.path.join(settings.ENCRYPTION_ROOT, t_id + '_private_key.pem')
    with open(prk_file, 'wb') as f:
        f.write(pem)

    # Save public key
    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    with open('public_key.pem', 'wb') as f:
        f.write(pem)

    # Load keys for encryption
    with open(prk_file, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend()
        )
    with open("public_key.pem", "rb") as key_file:
        public_key = serialization.load_pem_public_key(
            key_file.read(),
            backend=default_backend()
        )

    # Encrypt all items
    new_arr = []
    for i in message:
        if isinstance(i, str):
            i = i.encode('utf-8')
        encrypted = public_key.encrypt(
            i,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        new_arr.append(encrypted)

    return new_arr

def a_decryption(arr):
    # arr[1] is a Django FieldFile; convert to path string
    key_path = arr[1].path if hasattr(arr[1], 'path') else str(arr[1])
    
    # Load private key
    with open(key_path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend()
        )

    # Convert memoryview to bytes
    ciphertext_key = bytes(arr[0][1]) if isinstance(arr[0][1], memoryview) else arr[0][1]
    ciphertext_hash = bytes(arr[0][0]) if isinstance(arr[0][0], memoryview) else arr[0][0]

    # Decrypt
    key = private_key.decrypt(
        ciphertext_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    hash_id = private_key.decrypt(
        ciphertext_hash,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return [key, hash_id]
