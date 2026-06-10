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

// ── the transplant ──
// recBuf/donBuf: ArrayBuffers. Returns a new ArrayBuffer (valid VRM).
function transplantHair(recBuf,donBuf){
  const R=pGlb(recBuf), D=pGlb(donBuf);
  const rj=JSON.parse(JSON.stringify(R.json)); // deep clone, we mutate freely
  const dj=D.json;
  const recBin=new Uint8Array(recBuf,R.binOff);

  const hairMat=(json,mi)=>/hair/i.test(((json.materials||[])[mi]||{}).name||"");

  // every recipient bufferView keeps its original bytes
  const srcOf=new Map(); // bvIndex(in rj) → Uint8Array
  rj.bufferViews.forEach((bv,i)=>srcOf.set(i,recBin.subarray(bv.byteOffset||0,(bv.byteOffset||0)+bv.byteLength)));
  const addBv=(bytes)=>{rj.bufferViews.push({buffer:0,byteOffset:0,byteLength:bytes.length});srcOf.set(rj.bufferViews.length-1,bytes);return rj.bufferViews.length-1;};

  // ── 1. strip recipient hair (prims + hair-ish springs) ──
  rj.meshes.forEach(m=>{m.primitives=m.primitives.filter(p=>!hairMat(rj,p.material));});
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
  if(rsb&&rsb.springs){
    rsb.springs=rsb.springs.filter(s=>{
      const names=(s.joints||[]).map(jt=>(rj.nodes[jt.node]||{}).name||"");
      return !(/hair/i.test(s.name||"")||names.some(n=>/hair|J_Sec_Hair/i.test(n)));
    });
  }

  // ── 2. donor hair prims + their skins ──
  const donorMeshSkin={};
  dj.nodes.forEach(nd=>{if(nd.mesh!=null&&nd.skin!=null)donorMeshSkin[nd.mesh]=nd.skin;});
  const hairPrims=[]; // {prim, skinIdx}
  dj.meshes.forEach((m,mi)=>m.primitives.forEach(p=>{
    if(hairMat(dj,p.material))hairPrims.push({prim:p,skinIdx:donorMeshSkin[mi]});
  }));
  if(!hairPrims.length)throw new Error("ドナーに髪マテリアルのプリミティブが見つかりません");

  // ── 3. T = recHeadWorld · donorHeadWorld⁻¹ ──
  const dHm=pHumanoid(dj), rHm=pHumanoid(rj);
  if(dHm.head==null||rHm.head==null)throw new Error("humanoid head が見つかりません");
  const T=m4Mul(pNodeWorld(rj,rHm.head),m4Inv(pNodeWorld(dj,dHm.head)));
  const Tinv=m4Inv(T);
  const donorRole={};for(const[role,n]of Object.entries(dHm))donorRole[n]=role;

  // ── 4. copy needed donor nodes (non-role joints + ancestors up to a role) ──
  const dParent=new Array(dj.nodes.length).fill(-1);
  dj.nodes.forEach((nd,i)=>(nd.children||[]).forEach(c=>dParent[c]=i));
  const skinIdxs=[...new Set(hairPrims.map(h=>h.skinIdx))].filter(s=>s!=null);
  const toCopy=new Set();
  skinIdxs.forEach(si=>dj.skins[si].joints.forEach(jn=>{
    let c=jn;
    while(c>=0&&donorRole[c]==null&&!toCopy.has(c)){toCopy.add(c);c=dParent[c];}
  }));
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
    if(attach==null)attach=rHm.head;       // orphan chains hang off the head
    (rj.nodes[attach].children=rj.nodes[attach].children||[]).push(myNew);
  });

  // ── 5. rebuild each donor skin in recipient space ──
  const skinMap=new Map(); // donor skin idx → recipient skin idx
  skinIdxs.forEach(si=>{
    const sk=dj.skins[si];
    const joints=sk.joints.map(jn=>{
      if(nodeMap.has(jn))return nodeMap.get(jn);
      // role joint missing in recipient: degrade to head
      return rHm[donorRole[jn]]!=null?rHm[donorRole[jn]]:rHm.head;
    });
    const ibm=new Float32Array(pAccBytes(D,sk.inverseBindMatrices).buffer);
    const out=new Float32Array(ibm.length);
    for(let j=0;j<sk.joints.length;j++){
      const m=Array.from(ibm.subarray(j*16,j*16+16));
      out.set(m4Mul(m,Tinv),j*16);
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
  const accCopy=(ai,kind)=>{
    const a=dj.accessors[ai];
    let bytes=pAccBytes(D,ai);
    if(kind==="pos"||kind==="norm"){
      const f=new Float32Array(bytes.buffer,bytes.byteOffset,a.count*3);
      const mn=[1/0,1/0,1/0],mx=[-1/0,-1/0,-1/0];
      for(let i=0;i<a.count;i++){
        const v=kind==="pos"?m4Point(T,f[i*3],f[i*3+1],f[i*3+2]):m4Dir(T,f[i*3],f[i*3+1],f[i*3+2]);
        f[i*3]=v[0];f[i*3+1]=v[1];f[i*3+2]=v[2];
        for(let c=0;c<3;c++){mn[c]=Math.min(mn[c],v[c]);mx[c]=Math.max(mx[c],v[c]);}
      }
      const acc={bufferView:addBv(bytes),componentType:5126,count:a.count,type:"VEC3"};
      if(kind==="pos"){acc.min=mn;acc.max=mx;}
      rj.accessors.push(acc);return rj.accessors.length-1;
    }
    rj.accessors.push({bufferView:addBv(bytes),byteOffset:0,componentType:a.componentType,
      count:a.count,type:a.type,normalized:a.normalized});
    return rj.accessors.length-1;};

  rj.scenes=rj.scenes||[{nodes:[]}];
  hairPrims.forEach((h,k)=>{
    const a=h.prim.attributes;
    const attrs={POSITION:accCopy(a.POSITION,"pos")};
    if(a.NORMAL!=null)attrs.NORMAL=accCopy(a.NORMAL,"norm");
    if(a.TEXCOORD_0!=null)attrs.TEXCOORD_0=accCopy(a.TEXCOORD_0);
    if(a.JOINTS_0!=null)attrs.JOINTS_0=accCopy(a.JOINTS_0);
    if(a.WEIGHTS_0!=null)attrs.WEIGHTS_0=accCopy(a.WEIGHTS_0);
    const prim={attributes:attrs,indices:accCopy(h.prim.indices),material:copyMaterial(h.prim.material)};
    rj.meshes.push({name:"Hair_part_"+k,primitives:[prim]});
    rj.nodes.push({name:"Hair_part_"+k,mesh:rj.meshes.length-1,
      ...(h.skinIdx!=null&&skinMap.has(h.skinIdx)?{skin:skinMap.get(h.skinIdx)}:{})});
    rj.scenes[0].nodes.push(rj.nodes.length-1);
  });

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
