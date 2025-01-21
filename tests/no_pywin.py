import pytest

@pytest.fixture
def requirements_file():
    """Load the contents of the requirements.txt file."""
    try:
        with open("requirements.txt", "r") as file:
            return file.readlines()
    except FileNotFoundError:
        pytest.fail("requirements.txt file not found.")


def test_pywin_not_in_requirements(requirements_file):
    """Test to ensure 'pywin' is not present in requirements.txt."""
    forbidden_package = "pywin"
    # Check each line in the file for the forbidden package
    for line in requirements_file:
        assert forbidden_package not in line.strip(), f"'{forbidden_package}' should not be in requirements.txt"


