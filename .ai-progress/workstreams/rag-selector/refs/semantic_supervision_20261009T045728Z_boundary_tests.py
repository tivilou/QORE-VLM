from pathlib import Path
import json,sys,tempfile,subprocess,shutil,hashlib,argparse
p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--source-dir',required=True,type=Path);p.add_argument('--stage',required=True,type=Path);a=p.parse_args()
refs=a.root/'.ai-progress/workstreams/rag-selector/refs';stem='semantic_supervision_20261009T045728Z_boundary';script=refs/(stem+'_join.py');a.stage.mkdir(parents=True,exist_ok=False);commands=[]
def call(label,records,out,expected):
    cmd=[sys.executable,str(script),'--source-dir',str(a.source_dir),'--records-dir',str(records),'--output-dir',str(out)]
    r=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8');commands.append({'label':label,'command':cmd,'exit_status':r.returncode,'stdout':r.stdout,'stderr':r.stderr});assert r.returncode==expected,label
call('full_scope_readback',refs,a.stage/'valid',0)
for n,suffix in [('boundary_support_review.json','review.json'),('boundary_support_join.json','join.json')]:
    assert json.loads((a.stage/'valid'/n).read_text(encoding='utf-8'))==json.loads((refs/(stem+'_'+suffix)).read_text(encoding='utf-8'))
names={'partial_support_review.json':refs/'semantic_supervision_20261009T045728Z_partial_support_review.json','boundary_review_scope.json':refs/(stem+'_scope.json'),'new_judgments.txt':refs/(stem+'_judgments.txt')}
for kind in ('missing_scope_item','fake_quote','duplicate_judgment','missing_judgment','source_sha_mismatch'):
    d=a.stage/kind;d.mkdir()
    for n,src in names.items():shutil.copy2(src,d/n)
    if kind in ('missing_scope_item','source_sha_mismatch'):
        f=d/'boundary_review_scope.json';v=json.loads(f.read_text(encoding='utf-8'))
        if kind=='missing_scope_item':v['items'].pop()
        else:v['source_case_sha256']='0'*64
        f.write_text(json.dumps(v,ensure_ascii=False),encoding='utf-8')
    else:
        f=d/'new_judgments.txt';lines=f.read_text(encoding='utf-8').splitlines()
        if kind=='fake_quote':
            parts=lines[0].split('|');parts[2]='QUOTE_NOT_IN_SOURCE';lines[0]='|'.join(parts)
        elif kind=='duplicate_judgment':lines.append(lines[0])
        else:lines.pop()
        f.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    call(kind,d,a.stage/(kind+'_out'),1)
out={'checks':6,'passed':6,'semantic_truth_not_tested':True,'source_sha256':{n:hashlib.sha256((a.source_dir/n).read_bytes()).hexdigest() for n in ('case_study.json','support_review.json')},'commands':commands}
(a.stage/'verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'passed':6,'total':6,'summary':str(a.stage/'verification.json')}))
