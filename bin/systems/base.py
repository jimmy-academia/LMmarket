class BaseSystem:
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.test = data.get('test') or []
        self.city_data = self._collect_city_data()
        self.cities = list(self.city_data.keys())
        self.default_city = self._select_default_city()

    def _collect_city_data(self):
        ordered = {}
        for key, value in self.data.items():
            if key in ('test', 'user_loc'):
                continue
            if isinstance(value, dict):
                ordered[key] = value
        return ordered

    def _select_default_city(self):
        normalized = self.normalize_city(getattr(self.args, 'city', None))
        if normalized:
            return normalized
        for city in self.city_data:
            return city
        return None

    def normalize_city(self, city):
        if not city:
            return None
        key = city.strip().lower()
        if not key:
            return None
        if key in self.city_data:
            return key
        for name in self.city_data:
            if name.lower() == key:
                return name
        return None

    def get_city_data(self, city=None):
        key = self.normalize_city(city or self.default_city)
        if not key:
            return None
        return self.city_data.get(key)
