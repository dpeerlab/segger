"""
Parameter registry for extracting docstring descriptions and default values
from class constructors to populate a CLI (works with both Cyclopts and Typer).
"""

from typing import Any, Type, Annotated
from dataclasses import dataclass, MISSING
from docstring_parser import parse
import ast
import inspect
from pathlib import Path


@dataclass
class ParameterInfo:
    """Container for parameter information."""
    default: Any
    help: str
    type_annotation: Any
    _is_required: bool = False
    
    @property
    def is_required(self) -> bool:
        """Check if this parameter is required (has no default value)."""
        return self._is_required
    
    @property
    def has_default(self) -> bool:
        """Check if this parameter has a default value."""
        return not self._is_required


class ParameterRegistry:
    """
    Registry for collecting parameter information from multiple classes
    and making it available for CLI construction.
    
    Works with both Cyclopts and Typer frameworks.
    """
    
    def __init__(self, framework: str = "typer"):
        """
        Initialize the registry.
        
        Parameters
        ----------
        framework : str
            Either "typer" or "cyclopts" to specify which framework to use
        """
        self._parameters: dict[str, ParameterInfo] = {}
        self._registration_order: list[str] = []  # Track order of unprefixed names
        self._framework = framework.lower()
        
        if self._framework not in ("typer", "cyclopts"):
            raise ValueError("framework must be either 'typer' or 'cyclopts'")
    
    def register_from_file(self, file_path: str | Path, class_name: str, prefix: str | None = None) -> None:
        """
        Register a class by parsing its source file without importing.
        
        Parameters
        ----------
        file_path : str | Path
            Path to the Python source file containing the class
        class_name : str
            Name of the class to parse
        prefix : str, optional
            Optional prefix for parameter names (defaults to class_name)
            
        Raises
        ------
        ValueError
            If the class is not found or if parameters conflict
        FileNotFoundError
            If the file doesn't exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Parse the file
        with open(file_path, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        # Find the class definition
        class_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                class_node = node
                break
        
        if class_node is None:
            raise ValueError(f"Class '{class_name}' not found in {file_path}")
        
        # Extract docstring
        docstring = ast.get_docstring(class_node) or ""
        
        # Parse parameters from class body (for dataclasses) and __init__
        defaults, type_annotations = self._extract_from_ast(class_node)
        
        # Parse docstring
        doc_params = {}
        if docstring:
            parsed_doc = parse(docstring)
            doc_params = {param.arg_name: {
                'description': param.description or "",
                'type_name': param.type_name
            } for param in parsed_doc.params}
        
        # Use class name as prefix if not provided
        if prefix is None:
            prefix = class_name
        
        # Process each parameter
        self._process_parameters(defaults, type_annotations, doc_params, prefix)
    
    def register_class(self, cls: Type, prefix: str | None = None) -> None:
        """
        Register a class by inspecting it directly (requires importing).
        
        Parameters
        ----------
        cls : Type
            The class to register (e.g., LightningDataModule, LightningModule)
        prefix : str, optional
            Optional prefix for parameter names (defaults to class name)
            
        Raises
        ------
        ValueError
            If a parameter already exists with conflicting default value or description
        """
        from dataclasses import fields
        
        # Get docstring information
        docstring = parse(cls.__doc__ or "")
        doc_params = {param.arg_name: {
            'description': param.description or "",
            'type_name': param.type_name
        } for param in docstring.params}
        
        # Get default values and type annotations
        defaults = self._extract_defaults_from_class(cls)
        type_annotations = self._extract_type_annotations_from_class(cls)
        
        # Use class name as prefix if not provided
        if prefix is None:
            prefix = cls.__name__
        
        # Process each parameter
        self._process_parameters(defaults, type_annotations, doc_params, prefix)
    
    def _process_parameters(self, defaults: dict, type_annotations: dict, doc_params: dict, prefix: str) -> None:
        """Process and register parameters from extracted information."""
        for param_name, default_value in defaults.items():
            is_required = default_value is MISSING
            doc_info = doc_params.get(param_name, {'description': '', 'type_name': None})
            
            # Get type annotation (prefer from class annotations, fall back to docstring)
            type_ann = type_annotations.get(param_name)
            if type_ann is None and doc_info['type_name']:
                # Store the string representation from docstring if no annotation found
                type_ann = doc_info['type_name']
            
            param_info = ParameterInfo(
                default=None if is_required else default_value,
                help=doc_info['description'],
                type_annotation=type_ann,
                _is_required=is_required
            )
            
            # Register with both prefixed and unprefixed names
            prefixed_name = f"{prefix}.{param_name}"
            
            # Store prefixed version
            self._parameters[prefixed_name] = param_info
            
            # Check for conflicts on unprefixed name
            if param_name in self._parameters:
                self._check_conflicts(param_name, self._parameters[param_name], param_info)
                self._merge_info(param_name, param_info)
            else:
                # First time seeing this unprefixed name
                self._parameters[param_name] = param_info
                self._registration_order.append(param_name)
    
    def _extract_from_ast(self, class_node: ast.ClassDef) -> tuple[dict[str, Any], dict[str, str]]:
        """
        Extract defaults and type annotations from an AST ClassDef node.
        
        Returns
        -------
        tuple[dict[str, Any], dict[str, str]]
            (defaults_dict, type_annotations_dict)
        """
        defaults = {}
        type_annotations = {}
        
        # Look for annotated assignments in the class body (dataclass fields)
        for node in class_node.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                param_name = node.target.id
                
                # Get type annotation as string
                type_annotations[param_name] = ast.unparse(node.annotation)
                
                # Get default value if present
                if node.value is not None:
                    try:
                        # Try to evaluate simple literals
                        defaults[param_name] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        # For complex defaults, store the string representation
                        defaults[param_name] = ast.unparse(node.value)
                else:
                    defaults[param_name] = MISSING
            
            # Also look for __init__ method
            elif isinstance(node, ast.FunctionDef) and node.name == '__init__':
                # Extract parameters from __init__ signature
                for arg in node.args.args:
                    if arg.arg == 'self':
                        continue
                    
                    param_name = arg.arg
                    
                    # Get type annotation if present
                    if arg.annotation is not None:
                        type_annotations[param_name] = ast.unparse(arg.annotation)
                    
                    # Get default value if present
                    # Defaults are stored in reverse order at the end of args
                    num_defaults = len(node.args.defaults)
                    num_args = len(node.args.args) - 1  # Exclude self
                    arg_index = node.args.args.index(arg) - 1  # Exclude self from index
                    
                    if arg_index >= num_args - num_defaults:
                        # This arg has a default
                        default_index = arg_index - (num_args - num_defaults)
                        default_node = node.args.defaults[default_index]
                        try:
                            defaults[param_name] = ast.literal_eval(default_node)
                        except (ValueError, TypeError):
                            defaults[param_name] = ast.unparse(default_node)
                    else:
                        # No default - only add if not already from dataclass fields
                        if param_name not in defaults:
                            defaults[param_name] = MISSING
        
        return defaults, type_annotations
    
    def _extract_defaults_from_class(self, cls: Type) -> dict[str, Any]:
        """
        Extract parameter names and their default values from a class.
        
        Returns a dict mapping parameter names to their default values,
        or MISSING sentinel if no default exists.
        """
        from dataclasses import fields
        
        defaults = {}
        
        # Try dataclass fields first
        if hasattr(cls, '__dataclass_fields__'):
            for field in fields(cls):
                if field.default is not MISSING:
                    defaults[field.name] = field.default
                elif field.default_factory is not MISSING:
                    defaults[field.name] = field.default_factory()
                else:
                    defaults[field.name] = MISSING
        else:
            # Fall back to inspecting __init__ signature
            try:
                sig = inspect.signature(cls.__init__)
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    
                    if param.default is inspect.Parameter.empty:
                        defaults[param_name] = MISSING
                    else:
                        defaults[param_name] = param.default
            except (ValueError, TypeError):
                pass
        
        return defaults
    
    def _extract_type_annotations_from_class(self, cls: Type) -> dict[str, Any]:
        """
        Extract type annotations from a class.
        
        Returns a dict mapping parameter names to their type annotations.
        """
        from dataclasses import fields
        
        annotations = {}
        
        # Try dataclass fields first (they have type annotations)
        if hasattr(cls, '__dataclass_fields__'):
            for field in fields(cls):
                annotations[field.name] = field.type
        else:
            # Fall back to __init__ signature annotations
            try:
                sig = inspect.signature(cls.__init__)
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    
                    if param.annotation is not inspect.Parameter.empty:
                        annotations[param_name] = param.annotation
            except (ValueError, TypeError):
                pass
        
        return annotations
    
    def _check_conflicts(self, param_name: str, existing: ParameterInfo, new: ParameterInfo) -> None:
        """Check for conflicts between existing and new parameter info."""
        # Check default value conflict
        if existing.is_required != new.is_required:
            raise ValueError(
                f"Parameter '{param_name}' has conflicting requirements: "
                f"one class requires it, another has a default"
            )
        
        if not existing.is_required and existing.default != new.default:
            raise ValueError(
                f"Parameter '{param_name}' has conflicting default values: "
                f"{existing.default} vs {new.default}"
            )
        
        # Check description conflict (only if both are non-empty)
        if (existing.help and new.help and existing.help != new.help):
            raise ValueError(
                f"Parameter '{param_name}' has conflicting descriptions: "
                f"'{existing.help}' vs '{new.help}'"
            )
        
        # Check type annotation conflict (only if both are non-None)
        if (existing.type_annotation is not None and 
            new.type_annotation is not None and
            existing.type_annotation != new.type_annotation):
            raise ValueError(
                f"Parameter '{param_name}' has conflicting type annotations: "
                f"{existing.type_annotation} vs {new.type_annotation}"
            )
    
    def _merge_info(self, param_name: str, new: ParameterInfo) -> None:
        """Merge new parameter info with existing (prefer non-empty values)."""
        existing = self._parameters[param_name]
        
        # Merge descriptions (prefer non-empty)
        if new.help and not existing.help:
            self._parameters[param_name].help = new.help
        
        # Merge type annotations (prefer non-None)
        if new.type_annotation is not None and existing.type_annotation is None:
            self._parameters[param_name].type_annotation = new.type_annotation
    
    def get_parameter(self, param_name: str, **kwargs):
        """
        Get an Annotated type for use in Cyclopts function signatures.
        
        Only available when framework="cyclopts".
        
        Parameters
        ----------
        param_name : str
            The name of the parameter. Can be:
            - "param_name" - Returns the first registered parameter with this name
            - "ClassName.param_name" - Returns the parameter from specific class
        **kwargs
            Additional keyword arguments to pass to cyclopts.Parameter
            (e.g., validator, group, alias)
            
        Returns
        -------
        Annotated type
            An Annotated type with the parsed type and cyclopts.Parameter
            configured with help text and required status
            
        Raises
        ------
        ValueError
            If framework is not "cyclopts" or if parameter has no type annotation
        KeyError
            If the parameter has not been registered
        """
        if self._framework != "cyclopts":
            raise ValueError(
                "get_parameter() is only available when framework='cyclopts'. "
                "Use get() instead for Typer."
            )
        
        from cyclopts import Parameter
        
        # Check if it's a prefixed name first
        if '.' in param_name and param_name in self._parameters:
            info = self._parameters[param_name]
        elif param_name in self._parameters:
            info = self._parameters[param_name]
        else:
            raise KeyError(
                f"Parameter '{param_name}' has not been registered. "
                f"Available parameters: {', '.join(sorted(self._parameters.keys()))}"
            )
        
        if info.type_annotation is None:
            raise ValueError(
                f"Parameter '{param_name}' has no type annotation. "
                "Cannot create Annotated type."
            )
        
        # Create the Parameter with help and required, plus any user kwargs
        param_kwargs = dict(help=info.help, required=info.is_required)
        param_kwargs.update(kwargs)
        param = Parameter(**param_kwargs)
        
        # Return Annotated type
        return param
    
    def get_default(self, param_name: str) -> Any:
        """
        Get the default value for a parameter.
        
        Only available when framework="cyclopts".
        
        Parameters
        ----------
        param_name : str
            The name of the parameter
            
        Returns
        -------
        Any
            The default value (None if no default exists)
            
        Raises
        ------
        ValueError
            If framework is not "cyclopts"
        KeyError
            If the parameter has not been registered
        """
        if self._framework != "cyclopts":
            raise ValueError(
                "get_default() is only available when framework='cyclopts'. "
                "For Typer, use get() which returns the configured Option with the default."
            )
        
        if param_name not in self._parameters:
            raise KeyError(f"Parameter '{param_name}' has not been registered")
        
        return self._parameters[param_name].default
    
    def get(self, param_name: str, **kwargs):
        """
        Get a configured parameter for the CLI framework.
        
        - For Typer: Returns typer.Option configured with help and default
        - For Cyclopts: Use get_parameter() and get_default() instead
        
        Parameters
        ----------
        param_name : str
            The name of the parameter. Can be:
            - "param_name" - Returns the first registered parameter with this name
            - "ClassName.param_name" - Returns the parameter from specific class
        **kwargs
            Keyword arguments to pass to typer.Option (Typer only). Common kwargs:
            - default: Any - Override the default value from the registered class
            - alias: str - Short alias for the option (e.g., "-i")
            - exists: bool - For Path types, check if path exists
            - file_okay: bool - For Path types, allow files
            - dir_okay: bool - For Path types, allow directories
            - help: str - Override the help text
            
        Returns
        -------
        typer.Option (Typer mode)
            A Typer Option configured with help text and default value
            
        Raises
        ------
        ValueError
            If used with framework="cyclopts"
        KeyError
            If the parameter has not been registered
        """
        if self._framework == "cyclopts":
            raise ValueError(
                "get() is only available when framework='typer'. "
                "For Cyclopts, use get_parameter() and get_default() instead."
            )
        
        import typer
        
        # Check if it's a prefixed name first
        if '.' in param_name and param_name in self._parameters:
            info = self._parameters[param_name]
        elif param_name in self._parameters:
            info = self._parameters[param_name]
        else:
            raise KeyError(
                f"Parameter '{param_name}' has not been registered. "
                f"Available parameters: {', '.join(sorted(self._parameters.keys()))}"
            )
        
        # Use provided default or fall back to registered default
        if 'default' not in kwargs:
            # If required (no default), use ... as Typer's sentinel
            kwargs['default'] = ... if info.is_required else info.default
        
        # Use provided help or fall back to registered help
        if 'help' not in kwargs:
            kwargs['help'] = info.help
        
        # Create and return the Typer Option
        return typer.Option(**kwargs)
    
    def get_info(self, param_name: str) -> ParameterInfo:
        """
        Get the raw parameter information for a parameter.
        
        Parameters
        ----------
        param_name : str
            The name of the parameter (can be "ClassName.param_name" or just "param_name")
            
        Returns
        -------
        ParameterInfo
            Object containing default value and help text.
            - default: The default value (None if no default exists)
            - help: The help text (empty string if no description exists)
            - type_annotation: The type annotation (None if not found)
            - is_required: True if parameter has no default value
            - has_default: True if parameter has a default value
            
        Raises
        ------
        KeyError
            If the parameter has not been registered
        """
        if param_name not in self._parameters:
            raise KeyError(f"Parameter '{param_name}' has not been registered")
        
        return self._parameters[param_name]
    
    def get_parameter_names(self) -> list[str]:
        """
        Get a list of all registered parameter names.
        
        Returns
        -------
        list[str]
            List of parameter names (includes both prefixed and unprefixed versions)
        """
        return list(self._parameters.keys())
