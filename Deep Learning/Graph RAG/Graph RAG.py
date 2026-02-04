#LLM-based Entity–Relationship Extraction
def llm_json(system: str, user: str, max_retries: int=3) -> Dict:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}]
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            return result
            
        except json.JSONDecodeError as e:
            print("Error")
                
        except Exception as e:
            print("Error")
            
        if attempt < max_retries - 1:
            print("retry.")
    
    return {}



#Entity Extraction
ent_result = llm_json(system_prompt_entities, user_prompt_entities)
entities = ent_result.get("entities", []) if isinstance(ent_result, dict) else []

nodes: Dict[str, Dict] = {}

for item in entities:
    name = (item.get("name") or "").strip()
    etype = item.get("type") or "Other"
    
    if not name:
        continue
    
    if etype not in ENT_TYPES:
        etype = "Other"
    
    if name not in nodes:
        nodes[name] = {"label": etype}



#Relation Extraction
rel_result = llm_json(system_prompt_relations, user_prompt_relations)
rels = rel_result.get("relations", []) if isinstance(rel_result, dict) else []

edges: List[Dict] = []
invalid_count = 0

for r in rels:
    src = (r.get("src") or "").strip()
    dst = (r.get("dst") or "").strip()
    typ = (r.get("type") or "").strip()
    evid_doc = r.get("evidence_doc")
    evid_text = (r.get("evidence_text") or "").strip()
    is_explicit = r.get("is_explicit", True)
    confidence = r.get("confidence", 0.5)
    temporal_info = r.get("temporal_info")
    
    if not src or not dst or src==dst:
        invalid_count += 1
		
        continue
    
    if typ not in REL_TYPES:
        invalid_count += 1
		
        continue
    
    if src not in nodes:
        invalid_count += 1
		
        continue

    if dst not in nodes:
        invalid_count += 1
		
        continue
    
    if not isinstance(evid_doc, int) or not (0<=evid_doc<len(DOCS)):
        invalid_count += 1
		
        continue
   
    if not isinstance(confidence, (int, float)) or not (0.0<=confidence<=1.0):
        confidence = 0.5

    if not evid_text:
        evid_text = DOCS[evid_doc]
    
    edges.append({
        "src": src,
        "dst": dst,
        "type": typ,
        "evidence_doc": evid_doc,
        "evidence_text": evid_text,
        "is_explicit": is_explicit,
        "confidence": confidence,
        "temporal_info": temporal_info
    })