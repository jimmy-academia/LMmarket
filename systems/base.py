SPECIAL_KEYS = {"test", "user_loc"}


class BaseSystem:
    def __init__(self, args, data):
        self.args = args
        self.data = data
        test = data.get("test")
        self.test = test if test is not None else []
        user_loc = data.get("user_loc")
        self.user_loc = user_loc if user_loc is not None else {}
        self.result = {}
        self.city_lookup = {}
        self.city_list = []
        for key, payload in data.items():
            if key in SPECIAL_KEYS:
                continue
            if not isinstance(payload, dict):
                continue
            norm = key.strip().lower()
            if not norm or norm in self.city_lookup:
                continue
            self.city_lookup[norm] = key
            self.city_list.append(key)
        self.city_list.sort(key=lambda name: name.strip().lower())
        self.default_city = self.city_list[0] if self.city_list else None

    def list_cities(self):
        return list(self.city_list)

    def get_city_key(self, city=None):
        if city:
            if city in self.data and city not in SPECIAL_KEYS:
                return city
            norm = city.strip().lower()
            if norm in self.city_lookup:
                return self.city_lookup[norm]
        return self.default_city

    def get_city_data(self, city=None):
        key = self.get_city_key(city)
        if not key:
            return None
        return self.data.get(key)
