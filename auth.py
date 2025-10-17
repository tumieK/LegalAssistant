import streamlit as st
import hashlib
import json
import os

USER_DB = "users.json"


def hash_password(password: str) -> str:
    """Return SHA-256 hash of the password"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USER_DB):
        with open(USER_DB, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f, indent=2)


def register_user(username, password, location):
    users = load_users()
    if username in users:
        st.error("User already exists!")
        return False
    users[username] = {
        "password": hash_password(password),
        "location": location
    }
    save_users(users)
    st.success(f"User {username} registered successfully!")
    return True

def login_user(username, password):
    users = load_users()
    user = users.get(username)
    if user and user["password"] == hash_password(password):
        return True, user  
    return False, None
