// parts.js — VRM part transplant engine (hair first).
// Core idea: ONE transform T = recipientHeadWorld · donorHeadWorld⁻¹ maps the
// donor's hair into the recipient's space. Apply T to vertices and IBM⁻¹-side;
// then for every humanoid-role joint the new IBM collapses to the recipient's
// own inverse bind, and copied hair bones stay consistent. Joint order of the
// donor skin is preserved so JOINTS_0 buffers copy through untouched.
"use strict";

// ── mat4 (column-major, like glTF) ──
function m4Mul(a,b){const o=new Array(16);
  for(let c=0;c<4;c++)for(let r=0;r<4;r++){let v=0;for(let k=0;k<4;k++)v+=a[k*4+r]*b[c*4+k];o[c*4+r]=v;}
  return o;}
function m4FromTRS(t,q,s){
  const[x,y,z,w]=q;
  return [(1-2*(y*y+z*z))*s[0],2*(x*y+z*w)*s[0],2*(x*z-y*w)*s[0],0,
          2*(x*y-z*w)*s[1],(1-2*(x*x+z*z))*s[1],2*(y*z+x*w)*s[1],0,
          2*(x*z+y*w)*s[2],2*(y*z-x*w)*s[2],(1-2*(x*x+y*y))*s[2],0,
          t[0],t[1],t[2],1];}
function m4Inv(m){
  const inv=new Array(16);
  inv[0]=m[5]*m[10]*m[15]-m[5]*m[11]*m[14]-m[9]*m[6]*m[15]+m[9]*m[7]*m[14]+m[13]*m[6]*m[11]-m[13]*m[7]*m[10];
  inv[4]=-m[4]*m[10]*m[15]+m[4]*m[11]*m[14]+m[8]*m[6]*m[15]-m[8]*m[7]*m[14]-m[12]*m[6]*m[11]+m[12]*m[7]*m[10];
  inv[8]=m[4]*m[9]*m[15]-m[4]*m[11]*m[13]-m[8]*m[5]*m[15]+m[8]*m[7]*m[13]+m[12]*m[5]*m[11]-m[12]*m[7]*m[9];
  inv[12]=-m[4]*m[9]*m[14]+m[4]*m[10]*m[13]+m[8]*m[5]*m[14]-m[8]*m[6]*m[13]-m[12]*m[5]*m[10]+m[12]*m[6]*m[9];
  inv[1]=-m[1]*m[10]*m[15]+m[1]*m[11]*m[14]+m[9]*m[2]*m[15]-m[9]*m[3]*m[14]-m[13]*m[2]*m[11]+m[13]*m[3]*m[10];
  inv[5]=m[0]*m[10]*m[15]-m[0]*m[11]*m[14]-m[8]*m[2]*m[15]+m[8]*m[3]*m[14]+m[12]*m[2]*m[11]-m[12]*m[3]*m[10];
  inv[9]=-m[0]*m[9]*m[15]+m[0]*m[11]*m[13]+m[8]*m[1]*m[15]-m[8]*m[3]*m[13]-m[12]*m[1]*m[11]+m[12]*m[3]*m[9];
  inv[13]=m[0]*m[9]*m[14]-m[0]*m[10]*m[13]-m[8]*m[1]*m[14]+m[8]*m[2]*m[13]+m[12]*m[1]*m[10]-m[12]*m[2]*m[9];
  inv[2]=m[1]*m[6]*m[15]-m[1]*m[7]*m[14]-m[5]*m[2]*m[15]+m[5]*m[3]*m[14]+m[13]*m[2]*m[7]-m[13]*m[3]*m[6];
  inv[6]=-m[0]*m[6]*m[15]+m[0]*m[7]*m[14]+m[4]*m[2]*m[15]-m[4]*m[3]*m[14]-m[12]*m[2]*m[7]+m[12]*m[3]*m[6];
  inv[10]=m[0]*m[5]*m[15]-m[0]*m[7]*m[13]-m[4]*m[1]*m[15]+m[4]*m[3]*m[13]+m[12]*m[1]*m[7]-m[12]*m[3]*m[5];
  inv[14]=-m[0]*m[5]*m[14]+m[0]*m[6]*m[13]+m[4]*m[1]*m[14]-m[4]*m[2]*m[13]-m[12]*m[1]*m[6]+m[12]*m[2]*m[5];
  inv[3]=-m[1]*m[6]*m[11]+m[1]*m[7]*m[10]+m[5]*m[2]*m[11]-m[5]*m[3]*m[10]-m[9]*m[2]*m[7]+m[9]*m[3]*m[6];
  inv[7]=m[0]*m[6]*m[11]-m[0]*m[7]*m[10]-m[4]*m[2]*m[11]+m[4]*m[3]*m[10]+m[8]*m[2]*m[7]-m[8]*m[3]*m[6];
  inv[11]=-m[0]*m[5]*m[11]+m[0]*m[7]*m[9]+m[4]*m[1]*m[11]-m[4]*m[3]*m[9]-m[8]*m[1]*m[7]+m[8]*m[3]*m[5];
  inv[15]=m[0]*m[5]*m[10]-m[0]*m[6]*m[9]-m[4]*m[1]*m[10]+m[4]*m[2]*m[9]+m[8]*m[1]*m[6]-m[8]*m[2]*m[5];
  let det=m[0]*inv[0]+m[1]*inv[4]+m[2]*inv[8]+m[3]*inv[12];
  det=1.0/det;
  return inv.map(v=>v*det);}
function m4Point(m,x,y,z){return [m[0]*x+m[4]*y+m[8]*z+m[12], m[1]*x+m[5]*y+m[9]*z+m[13], m[2]*x+m[6]*y+m[10]*z+m[14]];}
function m4Dir(m,x,y,z){return [m[0]*x+m[4]*y+m[8]*z, m[1]*x+m[5]*y+m[9]*z, m[2]*x+m[6]*y+m[10]*z];}

// ── glb helpers (standalone copies; parts.js has no deps on the page) ──
function pGlb(buf){
  const dv=new DataView(buf);const total=dv.getUint32(8,true);
  let off=12,jsonStr=null,binOff=0;
  while(off+8<=total){const cl=dv.getUint32(off,true),ct=dv.getUint32(off+4,true);off+=8;
    if(ct===0x4E4F534A)jsonStr=new TextDecoder().decode(new Uint8Array(buf,off,cl));
    else if(ct===0x004E4942)binOff=off; off+=cl;}
  return {json:JSON.parse(jsonStr),binOff,buf};}
const pComp=ct=>ct===5126||ct===5125?4:ct===5123||ct===5122?2:1;
const pNum=t=>({SCALAR:1,VEC2:2,VEC3:3,VEC4:4,MAT4:16})[t];
// tight copy of an accessor's data (resolves byteStride / offsets)
function pAccBytes(g,ai){
  const a=g.json.accessors[ai],bv=g.json.bufferViews[a.bufferView];
  const nc=pNum(a.type),cs=pComp(a.componentType),elem=nc*cs;
  const st=bv.byteStride||elem;
  const base=g.binOff+(bv.byteOffset||0)+(a.byteOffset||0);
  const out=new Uint8Array(a.count*elem);
  const src=new Uint8Array(g.buf);
  for(let i=0;i<a.count;i++) out.set(src.subarray(base+i*st,base+i*st+elem),i*elem);
  return out;}
function pNodeWorld(json,ni){
  const parent=new Array(json.nodes.length).fill(-1);
  json.nodes.forEach((nd,i)=>(nd.children||[]).forEach(c=>parent[c]=i));
  let m=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
  const chain=[];let c=ni;
  while(c>=0){chain.unshift(c);c=parent[c];}
  chain.forEach(i=>{const nd=json.nodes[i];
    m=m4Mul(m,m4FromTRS(nd.translation||[0,0,0],nd.rotation||[0,0,0,1],nd.scale||[1,1,1]));});
  return m;}
function pHumanoid(json){
  const ext=json.extensions||{};const out={};
  if(ext.VRMC_vrm&&ext.VRMC_vrm.humanoid)
    for(const[n,b]of Object.entries(ext.VRMC_vrm.humanoid.humanBones||{}))out[n]=b.node;
  else if(ext.VRM&&ext.VRM.humanoid)
    (ext.VRM.humanoid.humanBones||[]).forEach(b=>{out[b.bone]=b.node;});
  return out;}

// ── category definitions ──
// test: material-name predicate; anchor: humanoid role the part rides on;
// morphs: carry morph targets + donor expression binds (face decals need it
// so blink/expressions keep working on the transplanted lashes/irises).
const PART_CATEGORIES={
  hair:  {test:n=>/hair/i.test(n), anchor:"head",  morphs:false, label:"髪"},
  face:  {test:n=>/(_FACE|_EYE|FaceBrow|FaceEyelash|FaceEyeline|EyeIris|EyeHighlight|EyeWhite|FaceMouth)/i.test(n)&&!/skin/i.test(n), anchor:"head", morphs:true, label:"顔"},
  cloth: {test:n=>/(cloth|onepiece|tops|bottoms|skirt|shoes|accessor)/i.test(n), anchor:"hips", morphs:false, label:"衣装"},
};
function transplantHair(recBuf,donBuf){return transplantParts(recBuf,donBuf,"hair");}

// ── the transplant ──
// recBuf/donBuf: ArrayBuffers. Returns a new ArrayBuffer (valid VRM).
function transplantParts(recBuf,donBuf,category,opts){
  const CAT=PART_CATEGORIES[category];
  if(!CAT)throw new Error("unknown category "+category);
  const addOnly=!!(opts&&opts.add);
  const R=pGlb(recBuf), D=pGlb(donBuf);
  const rj=JSON.parse(JSON.stringify(R.json)); // deep clone, we mutate freely
  const dj=D.json;
  const recBin=new Uint8Array(recBuf,R.binOff);

  const hairMat=(json,mi)=>CAT.test(((json.materials||[])[mi]||{}).name||"");

  // every recipient bufferView keeps its original bytes
  const srcOf=new Map(); // bvIndex(in rj) → Uint8Array
  rj.bufferViews.forEach((bv,i)=>srcOf.set(i,recBin.subarray(bv.byteOffset||0,(bv.byteOffset||0)+bv.byteLength)));
  const addBv=(bytes)=>{rj.bufferViews.push({buffer:0,byteOffset:0,byteLength:bytes.length});srcOf.set(rj.bufferViews.length-1,bytes);return rj.bufferViews.length-1;};

  // ── 1. strip recipient parts of this category (prims + matching springs) ──
  if(!addOnly)rj.meshes.forEach(m=>{
    const kept=m.primitives.filter(p=>!hairMat(rj,p.material));
    if(kept.length!==m.primitives.length&&kept.length&&m.primitives[0].targets){
      // glTF: all prims of a mesh share the weights array; keeping a subset is fine
    }
    m.primitives=kept;
  });
  const removedMesh=new Set();
  rj.meshes.forEach((m,i)=>{if(!m.primitives.length)removedMesh.add(i);});
  if(removedMesh.size){
    const remap=new Map();let k=0;
    rj.meshes.forEach((m,i)=>{if(!removedMesh.has(i))remap.set(i,k++);});
    rj.meshes=rj.meshes.filter((_,i)=>!removedMesh.has(i));
    rj.nodes.forEach(nd=>{
      if(nd.mesh!=null){ if(removedMesh.has(nd.mesh)){delete nd.mesh;delete nd.skin;} else nd.mesh=remap.get(nd.mesh); }
    });
  }
  const rsb=(rj.extensions||{}).VRMC_springBone;
  const springKill=addOnly?null:category==="hair"?/hair/i:category==="cloth"?/skirt|coat|cloth|sode|sleeve/i:null;
  if(rsb&&rsb.springs&&springKill){
    rsb.springs=rsb.springs.filter(s=>{
      const names=(s.joints||[]).map(jt=>(rj.nodes[jt.node]||{}).name||"");
      return !(springKill.test(s.name||"")||names.some(n=>springKill.test(n)));
    });
  }

  // ── 2. donor hair prims + their skins ──
  const donorMeshSkin={};
  dj.nodes.forEach(nd=>{if(nd.mesh!=null&&nd.skin!=null)donorMeshSkin[nd.mesh]=nd.skin;});
  const hairPrims=[]; // {prim, skinIdx, meshIdx}
  dj.meshes.forEach((m,mi)=>m.primitives.forEach(p=>{
    if(hairMat(dj,p.material))hairPrims.push({prim:p,skinIdx:donorMeshSkin[mi],meshIdx:mi});
  }));
  if(!hairPrims.length)throw new Error("ドナーに髪マテリアルのプリミティブが見つかりません");

  // ── 3. T = recHeadWorld · donorHeadWorld⁻¹ ──
  const dHm=pHumanoid(dj), rHm=pHumanoid(rj);
  const anchor=CAT.anchor;
  if(dHm[anchor]==null||rHm[anchor]==null)throw new Error("humanoid "+anchor+" が見つかりません");
  // Rest-pose bone ORIENTATIONS differ between models (e.g. unnormalized
  // VRM rest poses carry big rotations) while the posed geometry is upright
  // in both. So geometry moves by TRANSLATION ONLY, and each joint gets an
  // exact per-joint inverse bind: ibm'(j) = R(j)^-1 · T · worldDon(j) · ibm(j)
  // where R(j) is the joint's world AFTER attachment as the runtime sees it.
  const dA=pNodeWorld(dj,dHm[anchor]), rA=pNodeWorld(rj,rHm[anchor]);
  const T=[1,0,0,0, 0,1,0,0, 0,0,1,0, rA[12]-dA[12], rA[13]-dA[13], rA[14]-dA[14], 1];
  const donorRole={};for(const[role,n]of Object.entries(dHm))donorRole[n]=role;

  // ── 4. copy needed donor nodes (non-role joints + ancestors up to a role) ──
  const dParent=new Array(dj.nodes.length).fill(-1);
  dj.nodes.forEach((nd,i)=>(nd.children||[]).forEach(c=>dParent[c]=i));
  const skinIdxs=[...new Set(hairPrims.map(h=>h.skinIdx))].filter(s=>s!=null);
  // copy ONLY joints that real vertex weights touch (exporters often list the
  // whole armature in skin.joints — copying it all bloats the file and can
  // blow the runtime node budget)
  const usedJoints=new Map(); // skinIdx → Set(joint array index)
  skinIdxs.forEach(si=>usedJoints.set(si,new Set()));
  hairPrims.forEach(h=>{
    if(h.skinIdx==null)return;
    const a=h.prim.attributes;
    if(a.JOINTS_0==null||a.WEIGHTS_0==null)return;
    const J=pAccBytes(D,a.JOINTS_0), W=pAccBytes(D,a.WEIGHTS_0);
    const ja=dj.accessors[a.JOINTS_0], wa=dj.accessors[a.WEIGHTS_0];
    const jcs=pComp(ja.componentType), wct=wa.componentType;
    const jv=jcs===1?J:jcs===2?new Uint16Array(J.buffer,J.byteOffset,ja.count*4):new Uint32Array(J.buffer,J.byteOffset,ja.count*4);
    const wv=wct===5126?new Float32Array(W.buffer,W.byteOffset,wa.count*4)
            :wct===5123?new Uint16Array(W.buffer,W.byteOffset,wa.count*4):W;
    const set=usedJoints.get(h.skinIdx);
    for(let i=0;i<ja.count*4;i++)if(wv[i]>0)set.add(jv[i]);
  });
  const toCopy=new Set();
  skinIdxs.forEach(si=>{
    const used=usedJoints.get(si);
    dj.skins[si].joints.forEach((jn,k)=>{
      if(used.size&&!used.has(k))return;
      let c=jn;
      while(c>=0&&donorRole[c]==null&&!toCopy.has(c)){toCopy.add(c);c=dParent[c];}
    });
  });
  const nodeMap=new Map(); // donor node → recipient node index
  for(const[role,n]of Object.entries(dHm))if(rHm[role]!=null)nodeMap.set(n,rHm[role]);
  const order=[...toCopy].sort((a,b)=>a-b);
  order.forEach(dn=>{
    const nd=dj.nodes[dn];
    rj.nodes.push({name:(nd.name||"part")+"",translation:nd.translation,rotation:nd.rotation,scale:nd.scale});
    nodeMap.set(dn,rj.nodes.length-1);
  });
  // children wiring
  order.forEach(dn=>{
    const p=dParent[dn];
    const myNew=nodeMap.get(dn);
    let attach=null;
    if(p>=0&&nodeMap.has(p))attach=nodeMap.get(p);
    if(attach==null)attach=rHm[anchor];    // orphan chains hang off the anchor
    (rj.nodes[attach].children=rj.nodes[attach].children||[]).push(myNew);
  });

  // ── 5. rebuild each donor skin: per-joint exact inverse binds ──
  // R for copied nodes: runtime world = R(parent) · donorLocal(node);
  // for role-mapped nodes: R = recipient's own world.
  const dLocal=i=>{const nd=dj.nodes[i];return m4FromTRS(nd.translation||[0,0,0],nd.rotation||[0,0,0,1],nd.scale||[1,1,1]);};
  const Rmap=new Map(); // donor node idx → R (world after attachment)
  const recWorldCache=new Map();
  const recWorld=ni=>{if(!recWorldCache.has(ni))recWorldCache.set(ni,pNodeWorld(rj,ni));return recWorldCache.get(ni);};
  order.forEach(dn=>{
    const p=dParent[dn];
    let base;
    if(p>=0&&Rmap.has(p))base=Rmap.get(p);
    else if(p>=0&&donorRole[p]!=null&&rHm[donorRole[p]]!=null)base=recWorld(rHm[donorRole[p]]);
    else base=recWorld(rHm[anchor]);
    Rmap.set(dn,m4Mul(base,dLocal(dn)));
  });
  const Rof=jn=>{
    if(Rmap.has(jn))return Rmap.get(jn);
    const role=donorRole[jn];
    return recWorld(role!=null&&rHm[role]!=null?rHm[role]:rHm[anchor]);
  };
  const skinMap=new Map(); // donor skin idx → recipient skin idx
  skinIdxs.forEach(si=>{
    const sk=dj.skins[si];
    const joints=sk.joints.map(jn=>{
      if(nodeMap.has(jn))return nodeMap.get(jn);
      return rHm[donorRole[jn]]!=null?rHm[donorRole[jn]]:rHm[anchor];
    });
    const ibm=new Float32Array(pAccBytes(D,sk.inverseBindMatrices).buffer);
    const out=new Float32Array(ibm.length);
    for(let j=0;j<sk.joints.length;j++){
      const m=Array.from(ibm.subarray(j*16,j*16+16));
      const jn=sk.joints[j];
      const corr=m4Mul(m4Inv(Rof(jn)),m4Mul(T,pNodeWorld(dj,jn)));
      out.set(m4Mul(corr,m),j*16);
    }
    const bvi=addBv(new Uint8Array(out.buffer));
    rj.accessors.push({bufferView:bvi,componentType:5126,count:sk.joints.length,type:"MAT4"});
    rj.skins.push({joints,inverseBindMatrices:rj.accessors.length-1});
    skinMap.set(si,rj.skins.length-1);
  });

  // ── 6. copy materials/textures/images (dedup) ──
  const imgMap=new Map(),texMap=new Map(),matMap=new Map();
  const copyImage=ii=>{
    if(imgMap.has(ii))return imgMap.get(ii);
    const img=dj.images[ii];
    const bv=dj.bufferViews[img.bufferView];
    const bytes=new Uint8Array(donBuf,D.binOff+(bv.byteOffset||0),bv.byteLength).slice();
    rj.images=rj.images||[];
    rj.images.push({mimeType:img.mimeType,name:img.name,bufferView:addBv(bytes)});
    imgMap.set(ii,rj.images.length-1);return rj.images.length-1;};
  const copyTexture=ti=>{
    if(texMap.has(ti))return texMap.get(ti);
    const t=dj.textures[ti];
    rj.textures=rj.textures||[];
    rj.textures.push({source:copyImage(t.source),sampler:undefined});
    texMap.set(ti,rj.textures.length-1);return rj.textures.length-1;};
  const remapTexRefs=o=>{
    if(!o||typeof o!=="object")return;
    for(const k of Object.keys(o)){
      if(o[k]&&typeof o[k]==="object"){
        if(typeof o[k].index==="number"&&/texture/i.test(k))o[k]={...o[k],index:copyTexture(o[k].index)};
        else remapTexRefs(o[k]);
      }
    }};
  const copyMaterial=mi=>{
    if(matMap.has(mi))return matMap.get(mi);
    const m=JSON.parse(JSON.stringify(dj.materials[mi]));
    remapTexRefs(m);
    rj.materials.push(m);
    matMap.set(mi,rj.materials.length-1);return rj.materials.length-1;};

  // ── 7. geometry: copy accessors (POSITION transformed by T) ──
  const accCopy=(ai,kind,applyT)=>{
    const a=dj.accessors[ai];
    let bytes=pAccBytes(D,ai);
    if(kind==="pos"||kind==="norm"){
      const f=new Float32Array(bytes.buffer,bytes.byteOffset,a.count*3);
      const mn=[1/0,1/0,1/0],mx=[-1/0,-1/0,-1/0];
      for(let i=0;i<a.count;i++){
        let v=[f[i*3],f[i*3+1],f[i*3+2]];
        if(applyT)v=kind==="pos"?m4Point(T,v[0],v[1],v[2]):m4Dir(T,v[0],v[1],v[2]);
        f[i*3]=v[0];f[i*3+1]=v[1];f[i*3+2]=v[2];
        for(let c=0;c<3;c++){mn[c]=Math.min(mn[c],v[c]);mx[c]=Math.max(mx[c],v[c]);}
      }
      const acc={bufferView:addBv(bytes),componentType:5126,count:a.count,type:"VEC3",min:mn,max:mx};
      rj.accessors.push(acc);return rj.accessors.length-1;
    }
    rj.accessors.push({bufferView:addBv(bytes),byteOffset:0,componentType:a.componentType,
      count:a.count,type:a.type,normalized:a.normalized});
    return rj.accessors.length-1;};

  rj.scenes=rj.scenes||[{nodes:[]}];
  const donorMeshNewNode=new Map(); // donor mesh idx → [new node idx]
  hairPrims.forEach((h,k)=>{
    const a=h.prim.attributes;
    const skinned=h.skinIdx!=null&&skinMap.has(h.skinIdx);
    const attrs={POSITION:accCopy(a.POSITION,"pos",!skinned)};
    if(a.NORMAL!=null)attrs.NORMAL=accCopy(a.NORMAL,"norm",!skinned);
    if(a.TEXCOORD_0!=null)attrs.TEXCOORD_0=accCopy(a.TEXCOORD_0);
    if(a.JOINTS_0!=null)attrs.JOINTS_0=accCopy(a.JOINTS_0);
    if(a.WEIGHTS_0!=null)attrs.WEIGHTS_0=accCopy(a.WEIGHTS_0);
    const prim={attributes:attrs,indices:accCopy(h.prim.indices),material:copyMaterial(h.prim.material)};
    const mesh={name:CAT.label+"_part_"+k,primitives:[prim]};
    if(CAT.morphs&&h.prim.targets&&h.prim.targets.length){
      prim.targets=h.prim.targets.map(t=>{
        const out={};
        if(t.POSITION!=null&&!dj.accessors[t.POSITION].sparse)out.POSITION=accCopy(t.POSITION,"norm"); // deltas: rotate by T
        return out;
      });
      const names=(h.prim.extras&&h.prim.extras.targetNames)||(dj.meshes[h.meshIdx]&&dj.meshes[h.meshIdx].extras&&dj.meshes[h.meshIdx].extras.targetNames);
      if(names)mesh.extras={targetNames:names};
      mesh.weights=new Array(h.prim.targets.length).fill(0);
    }
    rj.meshes.push(mesh);
    rj.nodes.push({name:CAT.label+"_part_"+k,mesh:rj.meshes.length-1,
      ...(h.skinIdx!=null&&skinMap.has(h.skinIdx)?{skin:skinMap.get(h.skinIdx)}:{})});
    rj.scenes[0].nodes.push(rj.nodes.length-1);
    if(h.meshIdx!=null){
      if(!donorMeshNewNode.has(h.meshIdx))donorMeshNewNode.set(h.meshIdx,[]);
      donorMeshNewNode.get(h.meshIdx).push(rj.nodes.length-1);
    }
  });

  // ── 7b. expression binds for morph-bearing parts (face decals) ──
  if(CAT.morphs){
    const dEx=(((dj.extensions||{}).VRMC_vrm)||{}).expressions||{};
    const rEx=(((rj.extensions||{}).VRMC_vrm)||{}).expressions;
    if(rEx){
      for(const grp of ["preset","custom"]){
        for(const[name,ex]of Object.entries(dEx[grp]||{})){
          const tgt=(rEx[grp]||{})[name];
          if(!tgt)continue;
          (ex.morphTargetBinds||[]).forEach(b=>{
            const dm=dj.nodes[b.node]&&dj.nodes[b.node].mesh;
            if(dm==null||!donorMeshNewNode.has(dm))return;
            donorMeshNewNode.get(dm).forEach(nn=>{
              (tgt.morphTargetBinds=tgt.morphTargetBinds||[]).push({node:nn,index:b.index,weight:b.weight});
            });
          });
        }
      }
    }
  }

  // ── 8. donor hair springs + name-role-mapped colliders ──
  const dsb=(dj.extensions||{}).VRMC_springBone;
  if(dsb&&rsb){
    const colMap=new Map(),grpMap=new Map();
    const copyCollider=ci=>{
      if(colMap.has(ci))return colMap.get(ci);
      const c=dsb.colliders[ci];
      const tgt=nodeMap.has(c.node)?nodeMap.get(c.node):(donorRole[c.node]&&rHm[donorRole[c.node]]!=null?rHm[donorRole[c.node]]:null);
      if(tgt==null)return null;
      rsb.colliders=rsb.colliders||[];
      rsb.colliders.push({node:tgt,shape:JSON.parse(JSON.stringify(c.shape))});
      colMap.set(ci,rsb.colliders.length-1);return rsb.colliders.length-1;};
    const copyGroup=gi=>{
      if(grpMap.has(gi))return grpMap.get(gi);
      const g=dsb.colliderGroups[gi];
      const cs=(g.colliders||[]).map(copyCollider).filter(x=>x!=null);
      rsb.colliderGroups=rsb.colliderGroups||[];
      rsb.colliderGroups.push({name:g.name,colliders:cs});
      grpMap.set(gi,rsb.colliderGroups.length-1);return rsb.colliderGroups.length-1;};
    (dsb.springs||[]).forEach(s=>{
      const joints=(s.joints||[]);
      if(!joints.length||!joints.every(jt=>nodeMap.has(jt.node)))return; // hair springs only
      rsb.springs.push({name:s.name,
        joints:joints.map(jt=>({...jt,node:nodeMap.get(jt.node)})),
        colliderGroups:(s.colliderGroups||[]).map(copyGroup).filter(x=>x!=null)});
    });
  }

  // ── 9. assemble GLB ──
  const parts=[];let cur=0;
  rj.bufferViews.forEach((bv,i)=>{
    const src=srcOf.get(i);
    bv.byteOffset=cur;bv.byteLength=src.length;  // keep byteStride: original views may be interleaved
    parts.push(src);cur+=src.length;
    const pad=(4-(cur%4))%4;if(pad){parts.push(new Uint8Array(pad));cur+=pad;}
  });
  if(rj.buffers&&rj.buffers[0])rj.buffers[0].byteLength=cur;
  let jb=new TextEncoder().encode(JSON.stringify(rj));
  const jpad=(4-(jb.length%4))%4;
  if(jpad){const j2=new Uint8Array(jb.length+jpad);j2.set(jb);j2.fill(0x20,jb.length);jb=j2;}
  const totalLen=12+8+jb.length+8+cur;
  const out=new ArrayBuffer(totalLen);const odv=new DataView(out);const ou=new Uint8Array(out);
  odv.setUint32(0,0x46546C67,true);odv.setUint32(4,2,true);odv.setUint32(8,totalLen,true);
  odv.setUint32(12,jb.length,true);odv.setUint32(16,0x4E4F534A,true);ou.set(jb,20);
  let p=20+jb.length;
  odv.setUint32(p,cur,true);odv.setUint32(p+4,0x004E4942,true);p+=8;
  for(const part of parts){ou.set(part,p);p+=part.length;}
  return out;
}


// ── extraction: carve a category out of a VRM into a standalone PART file ──
// A part file is itself a minimal valid VRM: full skeleton + humanoid map
// (so transplantParts can role-map it later) + ONLY this category's prims,
// materials, textures, springs. Everything unreferenced is pruned, so a hair
// part is a few MB instead of the donor's 20.
function stripCategories(buf,categories,keepInverse){
  const G=pGlb(buf);
  const rj=JSON.parse(JSON.stringify(G.json));
  const bin=new Uint8Array(buf,G.binOff);
  const tests=categories.map(c=>PART_CATEGORIES[c].test);
  const inCat=mi=>tests.some(t=>t(((rj.materials||[])[mi]||{}).name||""));

  rj.meshes.forEach(m=>{m.primitives=m.primitives.filter(p=>keepInverse?inCat(p.material):!inCat(p.material));});
  const removedMesh=new Set();
  rj.meshes.forEach((m,i)=>{if(!m.primitives.length)removedMesh.add(i);});
  const meshRemap=new Map();let mk=0;
  rj.meshes.forEach((m,i)=>{if(!removedMesh.has(i))meshRemap.set(i,mk++);});
  rj.meshes=rj.meshes.filter((_,i)=>!removedMesh.has(i));
  rj.nodes.forEach(nd=>{
    if(nd.mesh!=null){ if(removedMesh.has(nd.mesh)){delete nd.mesh;delete nd.skin;} else nd.mesh=meshRemap.get(nd.mesh); }
  });

  // springs: keep the ones whose joints look like they belong to what we kept
  const sb=(rj.extensions||{}).VRMC_springBone;
  if(sb&&sb.springs){
    const want=k=>({hair:/hair|J_Sec_Hair/i,cloth:/skirt|coat|cloth|sode|sleeve/i,face:null})[k];
    const regs=categories.map(want).filter(Boolean);
    sb.springs=sb.springs.filter(s=>{
      const names=(s.joints||[]).map(jt=>(rj.nodes[jt.node]||{}).name||"").concat([s.name||""]);
      const hit=regs.some(r=>names.some(n=>r.test(n)));
      return keepInverse?hit:!hit;
    });
  }
  // expressions: drop binds to nodes without a mesh anymore
  const ex=(((rj.extensions||{}).VRMC_vrm)||{}).expressions;
  if(ex)for(const grp of ["preset","custom"])
    for(const e of Object.values(ex[grp]||{}))
      if(e.morphTargetBinds)e.morphTargetBinds=e.morphTargetBinds.filter(b=>rj.nodes[b.node]&&rj.nodes[b.node].mesh!=null);

  // ── reference closure → prune accessors / bufferViews / materials / textures / images / skins ──
  const usedAcc=new Set(),usedMat=new Set(),usedSkin=new Set();
  rj.meshes.forEach(m=>m.primitives.forEach(p=>{
    Object.values(p.attributes||{}).forEach(a=>usedAcc.add(a));
    if(p.indices!=null)usedAcc.add(p.indices);
    (p.targets||[]).forEach(t=>Object.values(t).forEach(a=>usedAcc.add(a)));
    if(p.material!=null)usedMat.add(p.material);
  }));
  rj.nodes.forEach(nd=>{if(nd.skin!=null)usedSkin.add(nd.skin);});
  const skinRemap=new Map();
  rj.skins=(rj.skins||[]).filter((sk,i)=>{
    if(!usedSkin.has(i))return false;
    skinRemap.set(i,skinRemap.size);
    if(sk.inverseBindMatrices!=null)usedAcc.add(sk.inverseBindMatrices);
    return true;});
  rj.nodes.forEach(nd=>{if(nd.skin!=null)nd.skin=skinRemap.get(nd.skin);});

  const usedTex=new Set();
  const scanTex=o=>{if(!o||typeof o!=="object")return;
    for(const k of Object.keys(o)){
      if(o[k]&&typeof o[k]==="object"){
        if(typeof o[k].index==="number"&&/texture/i.test(k))usedTex.add(o[k].index);
        else scanTex(o[k]);
      }}};
  const matRemap=new Map();
  rj.materials=(rj.materials||[]).filter((m,i)=>{
    if(!usedMat.has(i))return false;
    matRemap.set(i,matRemap.size);scanTex(m);return true;});
  rj.meshes.forEach(m=>m.primitives.forEach(p=>{if(p.material!=null)p.material=matRemap.get(p.material);}));

  const usedImg=new Set();const texRemap=new Map();
  rj.textures=(rj.textures||[]).filter((t,i)=>{
    if(!usedTex.has(i))return false;
    texRemap.set(i,texRemap.size);if(t.source!=null)usedImg.add(t.source);return true;});
  const fixTex=o=>{if(!o||typeof o!=="object")return;
    for(const k of Object.keys(o)){
      if(o[k]&&typeof o[k]==="object"){
        if(typeof o[k].index==="number"&&/texture/i.test(k))o[k].index=texRemap.get(o[k].index);
        else fixTex(o[k]);
      }}};
  rj.materials.forEach(fixTex);
  const imgRemap=new Map();
  rj.images=(rj.images||[]).filter((im,i)=>{
    if(!usedImg.has(i))return false;imgRemap.set(i,imgRemap.size);return true;});
  rj.textures.forEach(t=>{if(t.source!=null)t.source=imgRemap.get(t.source);});

  // VRM1 meta thumbnail references a texture — drop it (pruned)
  const vrmExt=(rj.extensions||{}).VRMC_vrm;
  if(vrmExt&&vrmExt.meta)delete vrmExt.meta.thumbnailImage;
  // VRM1 firstPerson/lookAt may reference removed meshes — drop for parts
  if(keepInverse&&vrmExt){delete vrmExt.firstPerson;delete vrmExt.lookAt;}

  // accessors: tight-copy kept ones; rebuild bufferViews fresh
  const accRemap=new Map();
  const newAcc=[];const newBv=[];const srcs=[];
  const G2={json:G.json,binOff:G.binOff,buf};
  [...usedAcc].sort((a,b)=>a-b).forEach(ai=>{
    const a=G.json.accessors[ai];
    const bytes=pAccBytes(G2,ai);
    newBv.push({buffer:0,byteOffset:0,byteLength:bytes.length});
    srcs.push(bytes);
    const na={bufferView:newBv.length-1,componentType:a.componentType,count:a.count,type:a.type};
    if(a.normalized)na.normalized=true;
    if(a.min)na.min=a.min;if(a.max)na.max=a.max;
    accRemap.set(ai,newAcc.length);newAcc.push(na);
  });
  rj.meshes.forEach(m=>m.primitives.forEach(p=>{
    for(const k of Object.keys(p.attributes))p.attributes[k]=accRemap.get(p.attributes[k]);
    if(p.indices!=null)p.indices=accRemap.get(p.indices);
    (p.targets||[]).forEach(t=>{for(const k of Object.keys(t))t[k]=accRemap.get(t[k]);});
  }));
  rj.skins.forEach(sk=>{if(sk.inverseBindMatrices!=null)sk.inverseBindMatrices=accRemap.get(sk.inverseBindMatrices);});
  // images get their own bufferViews
  rj.images.forEach(im=>{
    const bv=G.json.bufferViews[im.bufferView];
    const bytes=bin.subarray(bv.byteOffset||0,(bv.byteOffset||0)+bv.byteLength);
    newBv.push({buffer:0,byteOffset:0,byteLength:bytes.length});
    srcs.push(bytes);
    im.bufferView=newBv.length-1;
  });
  rj.accessors=newAcc;
  rj.bufferViews=newBv;

  // assemble
  const parts=[];let cur=0;
  rj.bufferViews.forEach((bv,i)=>{
    const src=srcs[i];
    bv.byteOffset=cur;bv.byteLength=src.length;
    parts.push(src);cur+=src.length;
    const pad=(4-(cur%4))%4;if(pad){parts.push(new Uint8Array(pad));cur+=pad;}
  });
  rj.buffers=[{byteLength:cur}];
  let jb=new TextEncoder().encode(JSON.stringify(rj));
  const jpad=(4-(jb.length%4))%4;
  if(jpad){const j2=new Uint8Array(jb.length+jpad);j2.set(jb);j2.fill(0x20,jb.length);jb=j2;}
  const totalLen=12+8+jb.length+8+cur;
  const out=new ArrayBuffer(totalLen);const odv=new DataView(out);const ou=new Uint8Array(out);
  odv.setUint32(0,0x46546C67,true);odv.setUint32(4,2,true);odv.setUint32(8,totalLen,true);
  odv.setUint32(12,jb.length,true);odv.setUint32(16,0x4E4F534A,true);ou.set(jb,20);
  let p=20+jb.length;
  odv.setUint32(p,cur,true);odv.setUint32(p+4,0x004E4942,true);p+=8;
  for(const part of parts){ou.set(part,p);p+=part.length;}
  return out;
}
// part file = only this category; base body = everything except all categories
function extractPart(buf,category){return stripCategories(buf,[category],true);}
function buildBase(buf){return stripCategories(buf,["hair","face","cloth"],false);}
