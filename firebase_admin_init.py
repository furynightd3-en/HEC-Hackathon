import os
import firebase_admin
from firebase_admin import credentials, firestore, auth


def init_firebase():
    """
    Uses GOOGLE_APPLICATION_CREDENTIALS to initialize Firebase Admin.
    For local dev: put serviceAccountKey.json in backend/ and set env accordingly.
    """
    if firebase_admin._apps:
        return firestore.client()

    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./serviceAccountKey.json")
    if not os.path.exists(cred_path):
        raise FileNotFoundError(
            f"Service account key not found at {cred_path}. "
            f"Download it from Firebase Console → Project settings → Service accounts."
        )
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    return firestore.client()


def verify_id_token(id_token: str):
    return auth.verify_id_token(id_token)
