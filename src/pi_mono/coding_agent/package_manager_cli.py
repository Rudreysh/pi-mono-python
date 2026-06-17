"""Package manager CLI commands (install/remove/list/update).

Ported from packages/coding-agent/src/package-manager-cli.ts (local-path subset).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

from pi_mono.config import APP_NAME, get_agent_dir
from pi_mono.coding_agent.core.package_manager import DefaultPackageManager
from pi_mono.core.settings_manager import SettingsManager

PackageCommand = Literal["install", "remove", "update", "list"]


@dataclass
class PackageCommandOptions:
    command: PackageCommand
    source: str | None = None
    local: bool = False
    help: bool = False
    invalid_option: str | None = None
    invalid_argument: str | None = None
    missing_option_value: str | None = None
    conflicting_options: str | None = None


def _get_package_command_usage(command: PackageCommand) -> str:
    if command == "install":
        return f"{APP_NAME} install <source> [-l]"
    if command == "remove":
        return f"{APP_NAME} remove <source> [-l]"
    if command == "update":
        return f"{APP_NAME} update"
    return f"{APP_NAME} list"


def _print_package_command_help(command: PackageCommand) -> None:
    if command == "install":
        print(
            f"""Usage:
  {_get_package_command_usage("install")}

Install a package and add it to settings.

Options:
  -l, --local    Install project-locally (.pi/settings.json)

Examples:
  {APP_NAME} install ./local/path
  {APP_NAME} install file:///path/to/package
  {APP_NAME} install npm:@scope/pkg
  {APP_NAME} install github.com/org/repo
"""
        )
        return

    if command == "remove":
        print(
            f"""Usage:
  {_get_package_command_usage("remove")}

Remove a package and its source from settings.
Alias: {APP_NAME} uninstall <source> [-l]

Options:
  -l, --local    Remove from project settings (.pi/settings.json)
"""
        )
        return

    if command == "update":
        print(
            f"""Usage:
  {_get_package_command_usage("update")}

Update pi and installed npm/git packages (not implemented in Python v1).
"""
        )
        return

    print(
        f"""Usage:
  {_get_package_command_usage("list")}

List installed packages from user and project settings.
"""
    )


def parse_package_command(args: list[str]) -> PackageCommandOptions | None:
    if not args:
        return None

    raw_command = args[0]
    if raw_command == "uninstall":
        command: PackageCommand | None = "remove"
    elif raw_command in ("install", "remove", "update", "list"):
        command = raw_command  # type: ignore[assignment]
    else:
        return None

    rest = args[1:]
    local = False
    help_requested = False
    invalid_option: str | None = None
    invalid_argument: str | None = None
    missing_option_value: str | None = None
    conflicting_options: str | None = None
    source: str | None = None

    index = 0
    while index < len(rest):
        arg = rest[index]
        if arg in ("-h", "--help"):
            help_requested = True
            index += 1
            continue

        if arg in ("-l", "--local"):
            if command in ("install", "remove"):
                local = True
            else:
                invalid_option = invalid_option or arg
            index += 1
            continue

        if arg.startswith("-"):
            invalid_option = invalid_option or arg
            index += 1
            continue

        if source is None:
            source = arg
        else:
            invalid_argument = invalid_argument or arg
        index += 1

    return PackageCommandOptions(
        command=command,
        source=source,
        local=local,
        help=help_requested,
        invalid_option=invalid_option,
        invalid_argument=invalid_argument,
        missing_option_value=missing_option_value,
        conflicting_options=conflicting_options,
    )


def _report_settings_errors(settings_manager: SettingsManager, context: str) -> None:
    for entry in settings_manager.drain_errors():
        scope = entry.get("scope", "unknown")
        error = entry.get("error")
        message = str(error)
        print(f"Warning ({context}, {scope} settings): {message}", file=sys.stderr)
        stack = getattr(error, "__traceback__", None)
        if stack is not None and hasattr(error, "__class__"):
            import traceback

            print(traceback.format_exc(), file=sys.stderr)


async def handle_package_command(args: list[str]) -> bool:
    options = parse_package_command(args)
    if options is None:
        return False

    if options.help:
        _print_package_command_help(options.command)
        return True

    if options.invalid_option:
        print(f'Unknown option {options.invalid_option} for "{options.command}".', file=sys.stderr)
        print(
            f'Use "{APP_NAME} --help" or "{_get_package_command_usage(options.command)}".',
            file=sys.stderr,
        )
        sys.exit(1)

    if options.missing_option_value:
        print(f"Missing value for {options.missing_option_value}.", file=sys.stderr)
        print(f"Usage: {_get_package_command_usage(options.command)}", file=sys.stderr)
        sys.exit(1)

    if options.invalid_argument:
        print(f"Unexpected argument {options.invalid_argument}.", file=sys.stderr)
        print(f"Usage: {_get_package_command_usage(options.command)}", file=sys.stderr)
        sys.exit(1)

    if options.conflicting_options:
        print(options.conflicting_options, file=sys.stderr)
        print(f"Usage: {_get_package_command_usage(options.command)}", file=sys.stderr)
        sys.exit(1)

    if options.command in ("install", "remove") and not options.source:
        print(f"Missing {options.command} source.", file=sys.stderr)
        print(f"Usage: {_get_package_command_usage(options.command)}", file=sys.stderr)
        sys.exit(1)

    cwd = os.getcwd()
    agent_dir = str(get_agent_dir())
    settings_manager = SettingsManager.create(cwd, agent_dir)
    _report_settings_errors(settings_manager, "package command")
    package_manager = DefaultPackageManager(
        cwd=cwd, agent_dir=agent_dir, settings_manager=settings_manager
    )

    package_manager.set_progress_callback(
        lambda event: (
            print(event.get("message", ""), file=sys.stderr)
            if event.get("type") == "start"
            else None
        )
    )

    try:
        if options.command == "install":
            assert options.source is not None
            await package_manager.install_and_persist(options.source, local=options.local)
            print(f"Installed {options.source}")
            return True

        if options.command == "remove":
            assert options.source is not None
            removed = await package_manager.remove_and_persist(options.source, local=options.local)
            if not removed:
                print(f"No matching package found for {options.source}", file=sys.stderr)
                sys.exit(1)
            print(f"Removed {options.source}")
            return True

        if options.command == "list":
            configured_packages = package_manager.list_configured_packages()
            user_packages = [pkg for pkg in configured_packages if pkg["scope"] == "user"]
            project_packages = [pkg for pkg in configured_packages if pkg["scope"] == "project"]

            if not configured_packages:
                print("No packages installed.")
                return True

            def format_package(pkg: dict[str, object]) -> None:
                source = str(pkg.get("source", ""))
                display = f"{source} (filtered)" if pkg.get("filtered") else source
                print(f"  {display}")
                installed_path = pkg.get("installedPath")
                if installed_path:
                    print(f"    {installed_path}")

            if user_packages:
                print("User packages:")
                for pkg in user_packages:
                    format_package(pkg)

            if project_packages:
                if user_packages:
                    print()
                print("Project packages:")
                for pkg in project_packages:
                    format_package(pkg)
            return True

        if options.command == "update":
            await package_manager.update(options.source)
            return True
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    return True
