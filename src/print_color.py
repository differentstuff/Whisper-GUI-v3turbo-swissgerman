import sys
from colorama import init, Fore, Style

# Initialize Colorama
init()


def print_header(text, color=Fore.CYAN):
    print(f"\n{color}=== {text} ==={Style.RESET_ALL}\n")


def print_info(text, color=Fore.WHITE):
    print(f"{color}{text}{Style.RESET_ALL}")


def print_success(text):
    print(f"{Fore.GREEN}> {text}{Style.RESET_ALL}")


def print_warning(text):
    print(f"{Fore.YELLOW}! {text}{Style.RESET_ALL}")


def print_error(text):
    print(f"{Fore.RED}X {text}{Style.RESET_ALL}")
