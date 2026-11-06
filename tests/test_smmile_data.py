"""Spatial prerequisite checks; these do not count SMMILe as a ready method."""
import ast
import copy
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml
from skimage.segmentation import slic

from milplus.cli import main
from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.io import digest_file
from milplus.smmile_data import (
    NICGrid,
    author_preprocessing,
    generate_superpixels,
    load_spatial_inputs,
    prepare_superpixels,
    validate_superpixels,
)
from milplus.upstream import verified_source


def reference():
    root, sha = verified_source('smmile')
    code = subprocess.check_output(['git', 'show', sha+':superpixel_generation.py'], cwd=root, text=True)
    names = {'get_nic_with_coord', 'generate_adjacency_matrix'}
    nodes = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    scope = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'independent/superpixel_generation.py', 'exec'), scope)  # noqa: S102 -- independent verified Git AST
    return scope


def inputs(dimension=4):
    rng = np.random.default_rng(11)
    xy = np.array([[x*256+512, y*256+768] for x in range(8) for y in range(5) if (x,y) not in {(2,2),(5,1),(0,0)}])
    rng.shuffle(xy)
    x = rng.normal(size=(len(xy), dimension)).astype(np.float32)
    return x, xy


@pytest.mark.parametrize('dimension', [1, 3, 4, 512, 1024, 1536])
def test_author_nic_grid_exact(dimension):
    x, xy = inputs(dimension)
    layout = NICGrid(xy, 256)
    original, mask, _ = reference()['get_nic_with_coord'](x, xy, 256, [])
    np.testing.assert_array_equal(layout.mask, mask)
    np.testing.assert_array_equal(layout.scatter(x), original)
    assert layout.mask[2,2] == 0 and layout.mask[5,1] == 0
    gathered = layout.scatter(x)[:,layout.mask == 1].T
    np.testing.assert_array_equal(layout.restore_rows(gathered), x)
    tx = torch.tensor(x, requires_grad=True)
    dense = layout.scatter(tx)
    recovered = layout.restore_rows(dense[:,layout.mask == 1].T)
    torch.testing.assert_close(recovered, tx, rtol=0, atol=0)
    recovered.square().sum().backward()
    torch.testing.assert_close(tx.grad, 2*tx, rtol=0, atol=0)


@pytest.mark.parametrize('dimension', [1, 3, 4, 16])
@pytest.mark.parametrize('compactness', [1., 50.])
def test_literal_author_slic_and_transposed_adjacency(dimension,compactness):
    x, xy = inputs(dimension)
    layout = NICGrid(xy, 256)
    ref = reference()
    normalized = (x-x.min())/(x.max()-x.min())*255
    grid, mask, _ = ref['get_nic_with_coord'](normalized, xy, 256, [])
    original = slic(grid.transpose(1,2,0), n_segments=max(10,int(len(x)/16)),
                    mask=mask, compactness=compactness, start_label=1).T
    adjacency = ref['generate_adjacency_matrix'](original)
    actual, adj = generate_superpixels(x, layout, compactness=compactness)
    np.testing.assert_array_equal(actual, original.T)
    np.testing.assert_array_equal(adj, adjacency)


def test_constant_features_have_explicit_finite_normalization():
    x, xy = inputs()
    a, b = generate_superpixels(np.ones_like(x), NICGrid(xy,256))
    assert np.isfinite(a).all() and (a[NICGrid(xy,256).mask == 1] > 0).all()
    validate_superpixels(a,b,NICGrid(xy,256))


@pytest.mark.parametrize('coords,stride,frame', [
    ([[0,0],[0,0]],1,'level0_pixels'), ([[0,0],[1.5,1]],1,'level0_pixels'),
    ([[0,0],[257,0]],256,'level0_pixels'), ([[0,0],[float('inf'),0]],1,'level0_pixels'),
    ([[0,0],[-1,0]],1,'level0_pixels'), ([[0,0],[0,1]],0,'level0_pixels'),
    ([[0,0],[0,1]],True,'level0_pixels'), ([[0,0],[0,1]],1,'unknown'),
    ([[0,0],[1+2j,0]],1,'level0_pixels'),
    ([],256,'level0_pixels'), ([[0,0],[1e30,0]],1,'level0_pixels'),
    ([[0,0],[1.1,0]],1,'micrometers')])
def test_invalid_or_impossible_lattice_rejected_before_allocation(coords,stride,frame):
    with pytest.raises(ValueError): NICGrid(np.asarray(coords),stride,frame,max_cells=100)


def test_micrometer_lattice_and_resource_caps():
    xy=np.array([[.25,.5],[.75,1.]])
    grid=NICGrid(xy,.5,'micrometers',max_cells=4,max_bytes=100)
    assert grid.shape==(2,2)
    with pytest.raises(ValueError,match='max_dense_bytes'): grid.scatter(np.ones((2,16),np.float32))
    with pytest.raises(ValueError,match='max_grid_cells'): NICGrid(np.array([[0,0],[5,5]]),1,max_cells=35)
    with pytest.raises(ValueError): NICGrid(xy,.5,'micrometers',max_cells=True)
    with pytest.raises(ValueError): NICGrid(xy,.5,'micrometers',max_bytes=0)


@pytest.mark.parametrize('kind',['negative','floating','wrong_shape','hole','zero_tissue','sparse_id','adj_shape','adj_float','adj_binary','adj_geometry'])
def test_superpixel_contract_rejects_invalid_assets(kind):
    x,xy=inputs(); layout=NICGrid(xy,256)
    sp,adj=generate_superpixels(x,layout)
    if kind=='negative':sp[0,0]=-1
    if kind=='floating':sp=sp.astype(float)
    if kind=='wrong_shape':sp=sp.T
    if kind=='hole':sp[layout.mask==0]=1
    if kind=='zero_tissue':sp[layout.mask==1]=0
    if kind=='sparse_id':sp[sp==sp.max()]=2**40
    if kind=='adj_shape':adj=adj[:-1]
    if kind=='adj_float':adj=adj.astype(float)
    if kind=='adj_binary':adj[0,0]=2
    if kind=='adj_geometry':adj[1,1]=1
    with pytest.raises(ValueError):validate_superpixels(sp,adj,layout)


def bundle_fixture(tmp_path):
    root=create_demo(tmp_path/'data',patients=18,dimension=4)
    b=Bundle(root)
    # The demo's registered physical lattice is synthetic and remains marked so.
    manifest=prepare_superpixels(b,tmp_path/'superpixels',256,provenance='synthetic grid for software testing only')
    task=yaml.safe_load((root/'task.yaml').read_text())
    task['encoders']['toy']['smmile']={'manifest':manifest}
    (root/'task.yaml').write_text(yaml.safe_dump(task))
    return Bundle(root), manifest


def test_bound_manifest_roundtrip_and_fingerprint(tmp_path):
    b,path=bundle_fixture(tmp_path)
    inv=b.check(deep=True)
    assert inv['features'][0]['smmile']['sha256']
    assert len(inv['features'])==len(b.rows)
    row=b.records('train')[0]
    x,_c,layout,sp,_adj=b.smmile.load(row)
    assert layout.restore_rows(layout.scatter(x)[:,layout.mask==1].T).equal(x)
    assert sp.shape==layout.shape
    m=json.loads(Path(path).read_text());m['preprocessing']+='; revised provenance'
    old=b.fingerprint
    with open(path,'w') as f:json.dump(m,f)
    assert Bundle(b.root).fingerprint!=old
    with pytest.raises(FileExistsError):prepare_superpixels(b, tmp_path/'superpixels',256,provenance='x')


@pytest.mark.parametrize('mutation',['features','coords','asset','asset_with_new_hash','fm','weights','dimension','source','synthetic','missing_slide','duplicate_slide','grid_shape','origin','count','no_provenance','shape_bomb'])
def test_asset_or_input_mutation_refused(tmp_path,mutation):
    b,path=bundle_fixture(tmp_path)
    m=json.loads(Path(path).read_text()); r=m['records'][0]
    row=b.rows.iloc[0].to_dict()
    asset=Path(path).parent/r['path']
    if mutation in {'features','coords'}:
        with h5py.File(b.root/row['path'],'r+') as h:
            key='features' if mutation=='features' else 'coords'
            values=h[key][:]; values[[0,1]]=values[[1,0]];h[key][:]=values
    elif mutation in {'asset','asset_with_new_hash'}:
        with h5py.File(asset,'r+') as h:h['superpixels'][0,0]=0
        if mutation.endswith('new_hash'):r['sha256']=digest_file(asset)
    elif mutation=='shape_bomb':
        with h5py.File(asset,'r+') as h:
            del h['adjacency'];h.create_dataset('adjacency',shape=(10**6,10**6),dtype='i8',chunks=(1,1))
        r['sha256']=digest_file(asset)
    elif mutation=='fm':m['fm']='other'
    elif mutation=='weights':m['weights']='other'
    elif mutation=='dimension':m['dimension']+=1
    elif mutation=='source':m['source_commit']='0'*40
    elif mutation=='synthetic':m['synthetic']=False
    elif mutation=='missing_slide':m['records'].pop()
    elif mutation=='duplicate_slide':m['records'][1]=copy.deepcopy(r)
    elif mutation=='grid_shape':r['grid_shape'][0]+=1
    elif mutation=='origin':r['origin'][0]+=1
    elif mutation=='count':r['superpixels']+=1
    elif mutation=='no_provenance':m['preprocessing']=''
    with open(path,'w') as f:json.dump(m,f)
    with pytest.raises(ValueError):Bundle(b.root).check(deep=True)


def test_cli_prepares_without_mutating_task_or_counting_method(tmp_path,capsys):
    data=create_demo(tmp_path/'data',patients=18,dimension=4)
    before=digest_file(data/'task.yaml')
    code=main(['prepare-smmile','--data',str(data),'--output',str(tmp_path/'assets'),
               '--patch-stride','256','--provenance','synthetic only'])
    assert code==0
    result=json.loads(capsys.readouterr().out)
    assert result['state']=='superpixel_assets_prepared'
    assert digest_file(data/'task.yaml')==before
    manifest=json.loads(Path(result['manifest']).read_text())
    assert manifest['synthetic'] is True and len(manifest['records'])==27


@pytest.mark.parametrize('kwargs',[{'patches_per_superpixel':0},{'patches_per_superpixel':True},
                                  {'compactness':0},{'compactness':float('nan')},{'compactness':True}])
def test_invalid_slic_parameters(kwargs):
    x,xy=inputs()
    with pytest.raises(ValueError):generate_superpixels(x,NICGrid(xy,256),**kwargs)


def test_sparse_geometry_adjacency_matches_author_transpose():
    mask=np.array([[1,0,1],[1,1,1]],np.uint8)
    coords=np.argwhere(mask)*256
    layout=NICGrid(coords,256)
    sp=np.array([[1,0,2],[1,2,2]])
    adj=author_preprocessing().generate_adjacency_matrix(sp.T)
    validate_superpixels(sp,adj,layout)
    assert adj[1,2]==1 and adj[2,1]==1


def test_stored_coordinate_precision_is_not_silently_lost(tmp_path):
    data=create_demo(tmp_path/'data',patients=18,dimension=4)
    b=Bundle(data);row=b.records('train')[0]
    with h5py.File(b.root/row['path'],'r+') as h:
        xy=h['coords'][:];xy[:,0]=2**25+np.arange(len(xy))*257
        h['coords'][:]=xy
    x,c=load_spatial_inputs(b,row)
    np.testing.assert_array_equal(c.numpy(),xy)
    layout=NICGrid(c.numpy(),257)
    assert layout.shape==(len(x),1)
    # Float32 would change this odd-valued, high-coordinate physical lattice.
    assert not np.array_equal(b.load(row)[1].numpy(),xy)
    with h5py.File(b.root/row['path'],'r+') as h:
        del h['coords']
        h['coords']=xy.astype(np.complex128)+2j
    with pytest.raises(ValueError,match='finite coordinates'):
        load_spatial_inputs(b,row)
