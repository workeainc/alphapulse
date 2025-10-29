#!/usr/bin/env python3
"""
AlphaPulse Backend Reorganization Analysis
Scans all files, extracts functions/classes, and identifies duplicates
"""

import os
import ast
import json
import logging
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FunctionInfo:
    """Information about a function"""
    name: str
    file_path: str
    line_number: int
    args: List[str]
    docstring: str
    decorators: List[str]
    is_async: bool
    is_method: bool
    class_name: str = ""

@dataclass
class ClassInfo:
    """Information about a class"""
    name: str
    file_path: str
    line_number: int
    methods: List[str]
    docstring: str
    bases: List[str]
    attributes: List[str]

@dataclass
class ImportInfo:
    """Information about imports"""
    module: str
    imports: List[str]
    file_path: str
    line_number: int

@dataclass
class FileAnalysis:
    """Analysis of a single file"""
    file_path: str
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    imports: List[ImportInfo]
    total_lines: int
    code_lines: int
    comment_lines: int
    empty_lines: int

@dataclass
class DuplicateAnalysis:
    """Analysis of duplicates"""
    duplicate_functions: Dict[str, List[FunctionInfo]]
    duplicate_classes: Dict[str, List[ClassInfo]]
    similar_files: List[Tuple[str, str, float]]  # file1, file2, similarity_score
    consolidation_recommendations: List[str]

class CodeAnalyzer:
    """Analyzes Python code for reorganization"""
    
    def __init__(self, backend_path: str = "."):
        self.backend_path = Path(backend_path)
        self.analysis_results: Dict[str, FileAnalysis] = {}
        self.duplicate_analysis = DuplicateAnalysis(
            duplicate_functions={},
            duplicate_classes={},
            similar_files=[],
            consolidation_recommendations=[]
        )
    
    def scan_all_files(self) -> Dict[str, FileAnalysis]:
        """Scan all Python files in the backend directory"""
        logger.info("Scanning all Python files...")
        
        for root, dirs, files in os.walk(self.backend_path):
            # Skip certain directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'htmlcov', '.benchmarks']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(self.backend_path)
                    
                    try:
                        analysis = self.analyze_file(file_path)
                        self.analysis_results[str(relative_path)] = analysis
                        logger.info(f"Analyzed: {relative_path}")
                    except Exception as e:
                        logger.error(f"Error analyzing {relative_path}: {e}")
        
        return self.analysis_results
    
    def analyze_file(self, file_path: Path) -> FileAnalysis:
        """Analyze a single Python file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count lines
        lines = content.split('\n')
        total_lines = len(lines)
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        empty_lines = sum(1 for line in lines if not line.strip())
        code_lines = total_lines - comment_lines - empty_lines
        
        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError:
            logger.warning(f"Syntax error in {file_path}, skipping AST analysis")
            return FileAnalysis(
                file_path=str(file_path),
                functions=[],
                classes=[],
                imports=[],
                total_lines=total_lines,
                code_lines=code_lines,
                comment_lines=comment_lines,
                empty_lines=empty_lines
            )
        
        # Extract functions, classes, and imports
        functions = self.extract_functions(tree, str(file_path))
        classes = self.extract_classes(tree, str(file_path))
        imports = self.extract_imports(tree, str(file_path))
        
        return FileAnalysis(
            file_path=str(file_path),
            functions=functions,
            classes=classes,
            imports=imports,
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            empty_lines=empty_lines
        )
    
    def extract_functions(self, tree: ast.AST, file_path: str) -> List[FunctionInfo]:
        """Extract function information from AST"""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = FunctionInfo(
                    name=node.name,
                    file_path=file_path,
                    line_number=node.lineno,
                    args=[arg.arg for arg in node.args.args],
                    docstring=ast.get_docstring(node) or "",
                    decorators=[self.get_decorator_name(d) for d in node.decorator_list],
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    is_method=False,
                    class_name=""
                )
                functions.append(func_info)
        
        return functions
    
    def extract_classes(self, tree: ast.AST, file_path: str) -> List[ClassInfo]:
        """Extract class information from AST"""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                attributes = []
                
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(item.name)
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                attributes.append(target.id)
                
                class_info = ClassInfo(
                    name=node.name,
                    file_path=file_path,
                    line_number=node.lineno,
                    methods=methods,
                    docstring=ast.get_docstring(node) or "",
                    bases=[self.get_base_name(base) for base in node.bases],
                    attributes=attributes
                )
                classes.append(class_info)
        
        return classes
    
    def extract_imports(self, tree: ast.AST, file_path: str) -> List[ImportInfo]:
        """Extract import information from AST"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_info = ImportInfo(
                        module=alias.name,
                        imports=[alias.asname or alias.name],
                        file_path=file_path,
                        line_number=node.lineno
                    )
                    imports.append(import_info)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                import_names = [alias.asname or alias.name for alias in node.names]
                import_info = ImportInfo(
                    module=module,
                    imports=import_names,
                    file_path=file_path,
                    line_number=node.lineno
                )
                imports.append(import_info)
        
        return imports
    
    def get_decorator_name(self, decorator: ast.expr) -> str:
        """Get decorator name from AST node"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self.get_decorator_name(decorator.value)}.{decorator.attr}"
        elif isinstance(decorator, ast.Call):
            return self.get_decorator_name(decorator.func)
        return "unknown"
    
    def get_base_name(self, base: ast.expr) -> str:
        """Get base class name from AST node"""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return f"{self.get_base_name(base.value)}.{base.attr}"
        return "unknown"
    
    def find_duplicates(self) -> DuplicateAnalysis:
        """Find duplicate functions and classes across files"""
        logger.info("Finding duplicates...")
        
        # Find duplicate functions
        function_groups: Dict[str, List[FunctionInfo]] = {}
        for file_path, analysis in self.analysis_results.items():
            for func in analysis.functions:
                key = func.name
                if key not in function_groups:
                    function_groups[key] = []
                function_groups[key].append(func)
        
        # Filter to only duplicates
        self.duplicate_analysis.duplicate_functions = {
            name: funcs for name, funcs in function_groups.items() 
            if len(funcs) > 1
        }
        
        # Find duplicate classes
        class_groups: Dict[str, List[ClassInfo]] = {}
        for file_path, analysis in self.analysis_results.items():
            for cls in analysis.classes:
                key = cls.name
                if key not in class_groups:
                    class_groups[key] = []
                class_groups[key].append(cls)
        
        # Filter to only duplicates
        self.duplicate_analysis.duplicate_classes = {
            name: classes for name, classes in class_groups.items() 
            if len(classes) > 1
        }
        
        # Find similar files (files with similar function/class names)
        self.find_similar_files()
        
        # Generate consolidation recommendations
        self.generate_consolidation_recommendations()
        
        return self.duplicate_analysis
    
    def find_similar_files(self):
        """Find files with similar content based on function/class names"""
        file_signatures = {}
        
        for file_path, analysis in self.analysis_results.items():
            # Create signature from function and class names
            func_names = {func.name for func in analysis.functions}
            class_names = {cls.name for cls in analysis.classes}
            signature = func_names.union(class_names)
            file_signatures[file_path] = signature
        
        # Compare files
        for i, (file1, sig1) in enumerate(file_signatures.items()):
            for file2, sig2 in list(file_signatures.items())[i+1:]:
                if sig1 and sig2:  # Skip empty signatures
                    intersection = len(sig1.intersection(sig2))
                    union = len(sig1.union(sig2))
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > 0.3:  # 30% similarity threshold
                        self.duplicate_analysis.similar_files.append((file1, file2, similarity))
    
    def generate_consolidation_recommendations(self):
        """Generate recommendations for file consolidation"""
        recommendations = []
        
        # Group files by prefix
        prefix_groups = {}
        for file_path in self.analysis_results.keys():
            if '/' in file_path:
                prefix = file_path.split('/')[-1].split('_')[0]
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append(file_path)
        
        # Recommend consolidation for groups with multiple files
        for prefix, files in prefix_groups.items():
            if len(files) > 1:
                recommendations.append(f"Consolidate {len(files)} files with prefix '{prefix}': {', '.join(files)}")
        
        # Recommend consolidation for similar files
        for file1, file2, similarity in self.duplicate_analysis.similar_files:
            if similarity > 0.5:
                recommendations.append(f"High similarity ({similarity:.1%}) between {file1} and {file2}")
        
        self.duplicate_analysis.consolidation_recommendations = recommendations
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_files": len(self.analysis_results),
            "total_functions": sum(len(analysis.functions) for analysis in self.analysis_results.values()),
            "total_classes": sum(len(analysis.classes) for analysis in self.analysis_results.values()),
            "total_imports": sum(len(analysis.imports) for analysis in self.analysis_results.values()),
            "duplicate_functions": len(self.duplicate_analysis.duplicate_functions),
            "duplicate_classes": len(self.duplicate_analysis.duplicate_classes),
            "similar_files": len(self.duplicate_analysis.similar_files),
            "file_analysis": {
                file_path: {
                    "total_lines": analysis.total_lines,
                    "code_lines": analysis.code_lines,
                    "functions": len(analysis.functions),
                    "classes": len(analysis.classes),
                    "imports": len(analysis.imports)
                }
                for file_path, analysis in self.analysis_results.items()
            },
            "duplicates": {
                "functions": {
                    name: [{"file": func.file_path, "line": func.line_number} for func in funcs]
                    for name, funcs in self.duplicate_analysis.duplicate_functions.items()
                },
                "classes": {
                    name: [{"file": cls.file_path, "line": cls.line_number} for cls in classes]
                    for name, classes in self.duplicate_analysis.duplicate_classes.items()
                }
            },
            "similar_files": [
                {"file1": f1, "file2": f2, "similarity": sim}
                for f1, f2, sim in self.duplicate_analysis.similar_files
            ],
            "recommendations": self.duplicate_analysis.consolidation_recommendations
        }
        
        return report

def main():
    """Main analysis function"""
    analyzer = CodeAnalyzer()
    
    # Scan all files
    analyzer.scan_all_files()
    
    # Find duplicates
    analyzer.find_duplicates()
    
    # Generate report
    report = analyzer.generate_report()
    
    # Save report
    with open("reorganization_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n=== AlphaPulse Reorganization Analysis ===")
    print(f"Total files analyzed: {report['total_files']}")
    print(f"Total functions: {report['total_functions']}")
    print(f"Total classes: {report['total_classes']}")
    print(f"Duplicate functions: {report['duplicate_functions']}")
    print(f"Duplicate classes: {report['duplicate_classes']}")
    print(f"Similar files: {report['similar_files']}")
    
    print(f"\n=== Consolidation Recommendations ===")
    for rec in report['recommendations']:
        print(f"- {rec}")
    
    print(f"\nDetailed report saved to: reorganization_analysis_report.json")

if __name__ == "__main__":
    main()
