
import ast, glob
files = glob.glob('**/*.py', recursive=True)
issues=[]
for f in files:
    if 'envMyIAModelChat' in f or f.startswith('.'):
        continue
    with open(f, 'r', encoding='utf-8') as fp:
        text = fp.read()
    try:
        tree = ast.parse(text, filename=f)
    except SyntaxError as e:
        issues.append((f,'syntax',str(e)))
        continue
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append((None,n.name,n.asname or n.name.split('.')[0],node.lineno))
        elif isinstance(node, ast.ImportFrom):
            module=node.module
            for n in node.names:
                if n.name=='*':
                    imports=[]
                    break
                imports.append((module,n.name,n.asname or n.name,node.lineno))
    if not imports:
        continue
    used=set()
    class NV(ast.NodeVisitor):
        def visit_Name(self,node):
            used.add(node.id)
    NV().visit(tree)
    for module,name,alias,lineno in imports:
        if alias not in used:
            issues.append((f,lineno,module,name,alias))
print('checked',len(files),'files')
for it in issues:
    f,line,module,name,alias=it
    print(f'{f}:{line} unused import: {alias} ({"from " + module if module else ""} {name})')
print('total',len(issues),'issues')
