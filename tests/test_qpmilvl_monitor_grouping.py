import ast
import inspect
import pytest
from milplus import qpmilvl_driver
from milplus.qpmilvl_source import original_root
from milplus.upstream import pinned_text


def expression(tree,name):
    nodes=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
    assert len(nodes)==1
    return compile(ast.Expression(nodes[0]),'<actual-monitor>','eval')


@pytest.mark.parametrize('loss,accuracy',[(1.1,1/3),(.7,2/3),(1.123456789,1/7),(3.141592653589793,2/9)])
def test_combined_monitor_preserves_literal_author_floating_point_grouping(loss,accuracy):
    root,commit=original_root()
    original=ast.parse(pinned_text(root,commit,'manager/manager.py'))
    # Locate the source expression by its assignment; it is not restated algebraically.
    branches=[n for n in ast.walk(original) if isinstance(n,ast.If) and ast.unparse(n.test)=="self.cfg['only_val_loss']"]
    assert len(branches)==1
    branch=branches[0]
    assert len(branch.body)==len(branch.orelse)==1
    assert ast.unparse(branch.body[0])=='monitor_metric = val_loss'
    native=expression(ast.Module(body=branch.orelse,type_ignores=[]),'monitor_metric')
    actual=expression(ast.parse(inspect.getsource(qpmilvl_driver.train)),'monitor')
    expected=eval(native,{'val_loss':loss,'val_acc':accuracy,'self':type('Context',(),{'cfg':{'only_val_loss':False}})()})
    assert eval(actual,{'val':{'loss':loss,'accuracy':accuracy}})==expected
