import hashlib
import crypt
import random
import string

# Input Data
data = [
    ("Lion", "Butterfly123!"),
    ("Tiger", "returnofthejedi"),
    ("Bear", "J@sonHouse"),
    ("Wolf", "sillywombat11"),
    ("Fox", "mi$tyHelp55"),
    ("Otter", "January2022"),
    ("Eagle", "P@$$w0rd"),
    ("Shark", "Ewug4"),
    ("Panda", "ieMuth6"),
    ("Falcon", "covidsucks")
]

def generate_shadow_line(user, password):
    # Salt for SHA-512 ($6$) consists of 8-16 random alphanumeric characters
    salt_charset = string.ascii_letters + string.digits
    salt = ''.join(random.choice(salt_charset) for _ in range(16))
    shadow_hash = crypt.crypt(password, f"$6${salt}")
    # Format: user:hash:last_change:min:max:warn:inactive:expire
    return f"{user.lower()}:{shadow_hash}:20386:0:99999:7:::"

def generate_ntlm_hash(password):
    # NTLM is MD4(UTF-16-LE(password))
    hash_obj = hashlib.new('md4', password.encode('utf-16le'))
    return hash_obj.hexdigest().upper()

def generate_pwdump_line(user, password, uid):
    ntlm = generate_ntlm_hash(password)
    # Format: user:uid:lm_hash:ntlm_hash:::
    lm_empty = "aad3b435b51404eeaad3b435b51404ee"
    return f"{user}:{uid}:{lm_empty}:{ntlm}:::"

# Execution
shadow_output = []
pwdump_output = []
start_uid = 1001

for i, (user, password) in enumerate(data):
    shadow_output.append(generate_shadow_line(user, password))
    pwdump_output.append(generate_pwdump_line(user, password, start_uid + i))

print("--- SHADOW FILE CONTENT ---")
print("\n".join(shadow_output))
print("\n--- PWDUMP FILE CONTENT ---")
print("\n".join(pwdump_output))
