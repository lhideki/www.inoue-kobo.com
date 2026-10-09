#!/usr/bin/env python3
"""Self-contained verified article estimator: four offline proxies, no provider API.
Main article scenario: o200k_base / compact_content_json.
Require pinned tiktoken 0.14.0 + locally hash-verified public encoding files.
"""
import argparse, ast, base64, hashlib, importlib.util, json, sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

def deny_network(event, args):
    if event in {"socket.connect", "socket.connect_ex", "socket.sendto", "socket.sendmsg", "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr", "urllib.Request", "http.client.connect", "http.client.send"}:
        raise RuntimeError("Offline estimator forbids network: " + event)

SOURCE_HASHES = {
 'jev': '16ddd989a329ed751feb7846fe22ba58cce81275d6f0979db96c4aee6ae27e42'}

DATA_HASHES = {
 'dataset.jsonl': '49295af0df7c14a408771b749924011f35902a82996126727cb44be911f18b91',
 'relation_choices.json': 'e5e00d938d20f8b32966fff7b17dc12d26fde687f11afb893009992586c4b70c',
 'jev-results-ja.jsonl': 'a41625f4ea8ce4355ae7e5074d9203312b4b1660f189489e3ebb726e0614c183',
 'jev-results-en.jsonl': '23c573654e42ba9b2597101cdeef5c36dab282a06162c569ab3ae307d949c2fc'}

def digest(value):
    return hashlib.sha256(value).hexdigest()

def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

def checked_source(path, kind):
    raw = path.read_bytes()
    if digest(raw) != SOURCE_HASHES[kind]:
        raise ValueError('Reviewed source hash changed: ' + kind)
    return ast.parse(raw)

def pure_namespace(nodes):
    """No imports, file access, provider calls, main or credential code may be executed."""
    tree = ast.Module(body=nodes, type_ignores=[])
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith, ast.Try)):
            raise ValueError('Non-pure AST')
        if isinstance(n, ast.Call):
            if not ((isinstance(n.func, ast.Name) and n.func.id in {'instructions', 'payload'}) or
                    (isinstance(n.func, ast.Attribute) and n.func.attr in {'dumps', 'encode'})):
                raise ValueError('Unapproved AST call')
        if isinstance(n, ast.Attribute) and n.attr.startswith('__'):
            raise ValueError('Private AST attribute')
    namespace = {'__builtins__': {}, 'json': json}
    exec(compile(ast.fix_missing_locations(tree), '<reviewed-pure-input-builders>', 'exec'), namespace)
    return namespace

def jev_builder(jev_path):
    jt = checked_source(jev_path, 'jev')
    main = next(n for n in jt.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    criteria = next(n for n in main.body if isinstance(n, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == 'criteria' for t in n.targets))
    loop = next(n for n in ast.walk(main) if isinstance(n, ast.For) and
                isinstance(n.target, ast.Tuple) and any(isinstance(x, ast.Name) and x.id == 'row' for x in n.target.elts))
    article = loop.body[0]
    branch = next(n for n in loop.body if isinstance(n, ast.If) and
                  isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Attribute) and n.test.left.attr == 'language')
    fn = ast.FunctionDef(name='jev_fields', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=x) for x in ('row','relations','args')], kwonlyargs=[], kw_defaults=[], defaults=[]),
                         body=[criteria, article, branch, ast.Return(value=ast.Tuple(elts=[ast.Name(id=x,ctx=ast.Load()) for x in ('state','instructions','criteria')],ctx=ast.Load()))], decorator_list=[])
    jf = pure_namespace([fn])['jev_fields']
    return lambda r,c,l: jf(r,c,SimpleNamespace(language=l))

def cost_fields(state, instructions, criteria):
    # Every candidate label AND description, repeated on every request.
    return [state, instructions] + [s for pair in criteria.items() for s in pair]

def cost_json(state, instructions, criteria):
    # Declared content serialization proxy, not Jev internal billing or captured wire bytes.
    return compact({'state':state,'questions':{'relation':{'type':'choice','criteria':criteria,'instructions':instructions}}})

def estimate(fields, encoder, mode):
    if mode == 'fieldwise':
        return sum(len(encoder.encode(s, disallowed_special=())) for s in cost_fields(*fields))
    if mode == 'compact_content_json':
        return len(encoder.encode(cost_json(*fields), disallowed_special=()))
    raise ValueError('Unknown serialization')

def price(tokens, per_million='0.042'):
    if tokens < 0:
        raise ValueError('Negative tokens')
    return str(Decimal(tokens)*Decimal(per_million)/Decimal(1000000))

ASSETS={
 'cl100k_base': '223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7',
 'o200k_base': '446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d'}

def local_encoder(name, asset):
    if name not in ASSETS: raise ValueError('Unsupported public proxy tokenizer')
    raw=asset.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ASSETS[name]: raise ValueError('Encoding asset hash mismatch')
    if importlib.util.find_spec('tiktoken') is None: raise RuntimeError('Missing pre-installed tiktoken; no installation attempted')
    import tiktoken
    if tiktoken.__version__ != "0.14.0":
        raise RuntimeError("Reproduction requires tiktoken 0.14.0")
    # Read the public pattern literal from installed source without executing constructors,
    # get_encoding, read_file, requests, environment reads or cache download logic.
    spec=importlib.util.find_spec('tiktoken_ext.openai_public')
    if spec is None or not spec.origin: raise RuntimeError('Missing public tokenizer definition source')
    source=Path(spec.origin).read_bytes()
    if digest(source) != "954392738e60d0fb6dca1dad80872efc47c8e2733babecbbf0a23970ed66c2cb":
        raise ValueError("Pinned public tokenizer definition hash mismatch")
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    returned=next(n.value for n in fn.body if isinstance(n,ast.Return) and isinstance(n.value,ast.Dict))
    expr=next(v for k,v in zip(returned.keys,returned.values) if isinstance(k,ast.Constant) and k.value=='pat_str')
    if isinstance(expr,ast.Name):
        expr=next(n.value for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==expr.id for t in n.targets))
    if isinstance(expr,ast.Call) and isinstance(expr.func,ast.Attribute) and isinstance(expr.func.value,ast.Constant) and expr.func.value.value=='|' and expr.func.attr=='join' and len(expr.args)==1:
        pattern='|'.join(ast.literal_eval(expr.args[0]))
    else: pattern=ast.literal_eval(expr)
    if not isinstance(pattern,str): raise ValueError('Unexpected public pattern AST; inspect before using')
    ranks={}
    for line in raw.splitlines():
        token,rank=line.split()
        token=base64.b64decode(token,validate=True)
        if token in ranks: raise ValueError('Duplicate BPE token')
        ranks[token]=int(rank)
    if set(ranks.values())!=set(range(len(ranks))): raise ValueError('Non-contiguous BPE ranks')
    enc=tiktoken.Encoding(name=name,pat_str=pattern,mergeable_ranks=ranks,special_tokens={})
    return enc, {'tiktoken_version':tiktoken.__version__,'public_definition_source_sha256':hashlib.sha256(source).hexdigest(),
                 'encoding_asset_sha256':ASSETS[name],'pattern_sha256':hashlib.sha256(pattern.encode()).hexdigest(),
                 'special_tokens':'empty; ordinary text encoding, no added special tokens'}

def main():
    p=argparse.ArgumentParser()
    for name in ('jev-source','data','cl100k-asset','o200k-asset','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise ValueError('Refuse overwrite')
    for name in ('dataset.jsonl','relation_choices.json'):
        if hashlib.sha256((a.data/name).read_bytes()).hexdigest()!=DATA_HASHES[name]: raise ValueError('Frozen data mismatch')
    jf=jev_builder(a.jev_source)
    rows=read_rows(a.data/'dataset.jsonl')
    relations=json.loads((a.data/'relation_choices.json').read_text(encoding='utf-8'))['relations']
    result={'status':'alternative_tokenizer_content_scenarios_only','requests_per_language':len(rows),
            'candidates_every_request':len(relations),'price_input_usd_per_million':'0.042','output_price_assumption':'free',
            'regional_multiplier_applied':False,'scenarios':[],
            'limitations':['Not measured Jev usage or historical cost','Not a Jev tokenizer','No guaranteed billing lower/upper bounds',
              'Unknown Jev internal formatting, system text, model, retries and rounding','Two languages use different article source content',
              'Content JSON excludes model routing and unknown internal wrappers; fieldwise excludes JSON keys/punctuation',
              'No caching discounts or batch discounts assumed; every candidate repeated on every request']}
    for name,path in [('o200k_base',a.o200k_asset),('cl100k_base',a.cl100k_asset)]:
        encoder,provenance=local_encoder(name,path)
        for mode in ('fieldwise','compact_content_json'):
            counts={l:sum(estimate(jf(r,relations,l),encoder,mode) for r in rows) for l in ('ja','en')}
            result['scenarios'].append({'tokenizer':name,'serialization':mode,'provenance':provenance,
              'proxy_input_tokens':counts|{'both':sum(counts.values())},
              'proxy_cost_usd':{l:price(n) for l,n in (counts|{'both':sum(counts.values())}).items()}})
    for lang in ('ja','en','both'):
        values=[s['proxy_input_tokens'][lang] for s in result['scenarios']]
        result.setdefault('scenario_sensitivity',{})[lang]={'min_scenario_tokens':min(values),'max_scenario_tokens':max(values),
            'max_minus_min_tokens':max(values)-min(values),'max_over_min_ratio':str(Decimal(max(values))/Decimal(min(values))),
            'interpretation':'range across four declared proxies, NOT billing bounds'}
    with a.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2); f.write('\n')
    a.output.chmod(0o600)
    print('Wrote aggregate proxy scenarios only')

if __name__ == "__main__":
    sys.addaudithook(deny_network)
    main()
