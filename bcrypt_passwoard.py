import bcrypt, getpass

pw = getpass.getpass("New password: ")
print(bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode())

