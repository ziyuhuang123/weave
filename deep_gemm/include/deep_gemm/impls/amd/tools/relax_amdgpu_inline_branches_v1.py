"""Add state-preserving scalar branch islands after LLVM MC macro expansion."""
import argparse,collections,hashlib,json,pathlib,re

def relax(source):
 lines=source.splitlines()
 for _lineno, _raw in enumerate(lines, 1):
  _s = _raw.split("//", 1)[0].split(";", 1)[0].strip()
  if re.match(r"^s_c?branch\S*\s+(?:[-+]?\d+|\d+[fb])\s*$", _s):
   raise RuntimeError(f"Line {_lineno}: raw numeric/directional branch unsupported across island insertion: {_s}")
 instruction=re.compile(r'^\s*([A-Za-z][\w.]*)\s*(.*?)\s*(?:;.*)?$')
 label=re.compile(r'^\s*([.$A-Za-z_][\w.$]*):')
 branch=re.compile(r'^(\s*s_(?:branch|cbranch_[a-z0-9_]+)\s+)([.$A-Za-z_][\w.$]*)(\s*(?:;.*)?)$')
 positions=[];labels={};ops=[];n=0
 for i,line in enumerate(lines):
  positions.append(n)
  m=label.match(line)
  if m:labels[m[1]]=i
  m=instruction.match(line)
  if m and not line.lstrip().startswith((';','//')):
   ops.append(i);n+=1
 checkpoints=[ops[j] for j in range(2500,len(ops),2500)]
 islands=collections.defaultdict(list);changes=[]
 for i,line in enumerate(lines):
  m=branch.match(line)
  if not m or m[2] not in labels:continue
  target=labels[m[2]]
  # 16 bytes per instruction is a conservative upper bound for this gfx942
  # expanded text. Use a 64-KiB threshold, half the signed branch byte range.
  if abs(positions[target]-positions[i])*16<=65536:continue
  points=[j for j in checkpoints if min(i,target)<j<max(i,target)]
  if target<i:points.reverse()
  assert points,(i,target)
  serial=len(changes)
  route=[f'.Lfi_far_{serial}_{k}' for k in range(len(points))]
  lines[i]=m[1]+route[0]+m[3]
  for k,point in enumerate(points):
   nxt=route[k+1] if k+1<len(route) else m[2]
   islands[point].append((route[k],nxt))
  changes.append({'line':i+1,'opcode':m[1].strip(),'target':m[2],
                  'estimated_max_bytes':abs(positions[target]-positions[i])*16,
                  'islands':len(points)})
 out=[]
 for i,line in enumerate(lines):
  if i in islands:
   after=f'.Lfi_island_after_{i}'
   out.append(f'\ts_branch {after}')
   for name,nxt in islands[i]:out.extend([name+':',f'\ts_branch {nxt}'])
   out.append(after+':')
  out.append(line)
 # Each original instruction remains in order, except the branch target now
 # reaches its original label through unconditional s_branch-only routes.
 assert all(len(route)>0 for route in islands.values())
 return '\n'.join(out)+'\n',{'changes':changes,'island_sites':len(islands),
   'added_s_branch':sum(len(v)+1 for v in islands.values()),
   'registers_added':0,'state_modifying_instructions_added':0,
   'source_sha256':hashlib.sha256(source.encode()).hexdigest()}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source',type=pathlib.Path);p.add_argument('dest',type=pathlib.Path)
 p.add_argument('--report',type=pathlib.Path,required=True);a=p.parse_args()
 assert not a.dest.exists() and not a.report.exists()
 result,report=relax(a.source.read_text());a.dest.write_text(result,newline='\n')
 report['dest_sha256']=hashlib.sha256(a.dest.read_bytes()).hexdigest()
 a.report.write_text(json.dumps(report,indent=2));print(json.dumps(report))
