class MenuEntry:
    def __init__(self, text, id):
        self.sub_menu = []
        self.text = text
        self.id = id
        self.sub_menu_id_counter = 0

    def add_sub(self, text):
        sub_entry = MenuEntry(text, self.sub_menu_id_counter)
        self.sub_menu_id_counter += 1
        self.sub_menu.append(sub_entry)
