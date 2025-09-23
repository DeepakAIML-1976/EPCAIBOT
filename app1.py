from modules import db_handler
print("Datasheets:", db_handler.get_all_datasheets())
print("Vendors:", db_handler.get_all_vendors())