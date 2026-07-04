class ServiceManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceManager, cls).__new__(cls)
            cls._instance._services = {}
        return cls._instance

    @classmethod
    def register(cls, service_class, service_instance):
        """Register a service instance for a specific interface/class type."""
        manager = cls()
        manager._services[service_class] = service_instance
        print(f"[ServiceManager] Registered service: {service_class.__name__}")

    @classmethod
    def get(cls, service_class):
        """Retrieve a registered service instance."""
        manager = cls()
        instance = manager._services.get(service_class)
        if instance is None:
            raise ValueError(f"Service '{service_class.__name__}' is not registered.")
        return instance
