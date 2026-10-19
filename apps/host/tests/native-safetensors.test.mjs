import test from 'node:test';
import assert from 'node:assert/strict';
import {checkRunConfig} from '../server/run-config-check.mjs';
import {resourceSettings,saferStart} from '../server/native-resources.mjs';
import {preferNativeCatalog,sourceServerEnvironment} from '../server/native-safetensors.mjs';

test('native source settings and diagnostics are recognized',()=>{
 const result=checkRunConfig(JSON.stringify({tokenizer_format:'safetensors',cache_diagnostics:true}));
 assert.equal(result.ok,true);assert.deepEqual(result.issues,[]);
});
test('native resource preference preserves fully resident checkpoint experts',()=>{
 const config={tokenizer_format:'safetensors',args:['--model','original-checkpoint']};
 const result=resourceSettings(config,{ram_gb:64,cpu:{cores:16},gpus:[{index:0,vram_gb:32}]},'work',{arenaGb:68});
 assert.equal(result.args.includes('--resident-budget-gib'),false);
 assert.equal(result.env?.STRATA_RESIDENT_HEADROOM_GIB,undefined);
 assert.equal(saferStart(config,{ram_gb:64},{arenaGb:68},'cannot allocate pinned memory'),null);
});
test('source app selects a prepared native profile without changing the generic catalog',()=>{
 const catalog={recommendation:{family:'qwen',model:'IQ3_S',download_gb:83.6,setup_command:'install GGUF'},families:[{family:'safetensors',sizes:[{id:'nvfp4-native',model:'NVFP4 Native'}]}]};
 assert.equal(preferNativeCatalog(catalog,{models:[]}),catalog);
 const preferred=preferNativeCatalog(catalog,{models:[{model:'nvfp4-native',ready:true,context:262144}]});
 assert.equal(preferred.recommendation.family,'safetensors');assert.equal(preferred.recommendation.context,262144);
 assert.equal(preferred.recommendation.download_gb,0);assert.equal(preferred.recommendation.setup_command,null);
 assert.equal(catalog.recommendation.family,'qwen');
});
test('source instance uses its private config key without an empty environment override',()=>{
 const original={STRATA_API_KEY:'ambient',PATH:'tools'};
 const env=sourceServerEnvironment(original);
 assert.equal(Object.hasOwn(env,'STRATA_API_KEY'),false);assert.equal(env.PATH,'tools');
 assert.equal(original.STRATA_API_KEY,'ambient');
});
