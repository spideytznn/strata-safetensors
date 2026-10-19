/** Prefer the prepared stable native profile in this fork's source app. */
export function preferNativeCatalog(catalog, installed) {
 const family=catalog?.families?.find(f=>f.family==='safetensors');
 const size=family?.sizes?.find(s=>s.id==='nvfp4-native');
 const model=installed?.models?.find(m=>m.model===size?.id&&m.ready);
 if(!size||!model)return catalog;
 return {...catalog,recommendation:{...catalog.recommendation,family:'safetensors',model:size.model,
  title:size.model,context:model.context??32768,low_ram:false,download_gb:0,setup_command:null,notes:[],
  why:'Existing original safetensors checkpoint; keeps the saved native profile.'}};
}

export function sourceServerEnvironment(base=process.env) {
 const env={...base,PYTHONUNBUFFERED:'1',PYTHONIOENCODING:'utf-8'};
 // The source host writes its fresh per-instance key to the private config.
 // An explicitly empty environment key now causes a 0.2 server startup error.
 delete env.STRATA_API_KEY;
 return env;
}
