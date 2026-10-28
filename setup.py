from setuptools import setup, find_packages

setup(
    name="smart_ev_charging_scheduler",
    version="1.0.0",
    packages=find_packages(include=["Chargym_Charging_Station*", "training*"]),
    install_requires=["gymnasium>=0.29", "numpy", "scipy"],
)
