import math


class Paginator:

    def __init__(self, items, page=1, page_limit=30):
        self.__items = items
        self.__page = page
        self.__page_limit = page_limit

    def hits(self):
        return len(self.__items)

    def page_limit(self):
        return self.__page_limit

    def cur_page(self):
        return self.__page

    def num_pages(self):
        return math.ceil(len(self.__items) / self.__page_limit)

    def has_items_on_this_page(self):
        return 0 < self.__page <= self.num_pages()

    def has_next_page(self):
        return self.__page < self.num_pages()

    def next_page(self):
        if self.has_next_page():
            return max(1, self.__page + 1)
        else:
            return None

    def has_prev_page(self):
        return self.__page > 1 and self.num_pages() > 1

    def prev_page(self):
        if self.has_next_page():
            return min(self.num_pages(), self.__page - 1)
        else:
            return None

    def export_page_items(self):
        if self.has_items_on_this_page():
            return self.__items[
                   self.__page_limit * (self.__page - 1):
                   min(len(self.__items), self.__page_limit * self.__page)
                   ]
        else:
            return []
