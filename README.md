# DEC: Discrete Exterior Calculus in MATLAB

`DEC.m` is a single MATLAB class for discrete exterior calculus on triangle meshes
(oriented 2-manifolds: closed or with boundary, any genus, one or several components).
It includes a Helmholtz–Hodge decomposition of tangent vector fields that also returns
the two **scalar potentials**: the divergence potential α and the rotation (stream)
potential β.

It needs only base MATLAB (R2016b or newer, no toolboxes) and also runs in GNU Octave 8.

## Quick start

```matlab
[V, F] = DEC.icosphere(4);              % or your own mesh (nV x 3, nF x 3)
M = DEC(V, F);

X = randn(M.nV, 3);                     % any per-vertex field (projected implicitly)
[XE, XC, XH, alpha, beta, info] = M.decomposeVectorField(X, 'vertex');

M.plotScalar(alpha);                    % divergence potential on vertices
figure; M.plotScalar(beta);             % stream potential on faces
hold on; M.plotField(XC, 2);            % divergence-free part, per face
```

Working directly with 1-forms:

```matlab
omega = M.vectorFieldToOneForm(X, 'vertex');            % "flat"
[wE, wC, wH, alpha, beta, info] = M.hodgeDecomposition(omega);
XE = M.oneFormToFaceVectors(wE);                       % "sharp" (Whitney)
```

Run `test_DEC` (or `test_DEC(true)` for plots) to verify everything.

## Discrete forms and operators

Edges are stored once, oriented from the lower to the higher vertex index (`M.E`).

| object | size | meaning |
|---|---|---|
| 0-form | nV × 1 | values at vertices |
| 1-form | nE × 1 | integral along each oriented edge |
| 2-form | nF × 1 | integral over each face |
| dual 0-form | nF × 1 | value at each circumcentre (dual vertex) |

| operator | size | definition |
|---|---|---|
| `d0` | nE × nV | signed incidence edge–vertex (gradient) |
| `d1` | nF × nE | signed incidence face–edge (curl), `d1*d0 = 0` |
| `star0` | nV × nV | dual cell areas: mixed Voronoi (default) or `'barycentric'` |
| `star1` | nE × nE | cotan weights \|e*\|/\|e\| = (cot a + cot b)/2 |
| `star2` | nF × nF | 1 / face area |

Further methods: `laplacian0` (cotan Laplacian `d0'*star1*d0`), `laplacian2dual`,
`laplacian1` (Hodge Laplacian on 1-forms), `divergence`, `curl`, `innerProduct1`,
`harmonicDimension`, `gradient0`, `faceToVertex`, `vertexToFace`, `plotScalar`,
`plotField`, and the test meshes `DEC.icosphere`, `DEC.diskMesh`, `DEC.torusMesh`.

## Vector fields and 1-forms

* **Flat** (`vectorFieldToOneForm`). Per-vertex field: midpoint rule
  ω_ij = ½(X_i + X_j)·(v_j − v_i). Per-face field: average of X_f·e over the faces next to the edge.
* **Sharp** (`oneFormToFaceVectors`). Whitney interpolation evaluated at the barycentre:
  X_f = ⅓ Σ ω_ij (∇φ_j − ∇φ_i). It is exact for piecewise-constant fields, and for
  ω = d0 f it returns exactly the gradient of the linear interpolant of f.
  `oneFormToVertexVectors` averages these to the vertices for plotting.

## Hodge decomposition and potentials

Any 1-form splits uniquely and orthogonally (in the star1 inner product ⟨a,b⟩ = aᵀ·star1·b) as

```
omega  =  d0*alpha   +   ( -inv(star1)*d1'*beta )   +   h
          exact          co-exact                       harmonic
          curl-free      divergence-free                curl- and divergence-free
          X ~ grad α     X ~ n × grad β
```

The two potentials come from two sparse Poisson problems:

1. **Divergence potential α (vertex 0-form).** Apply `d0'*star1` (a discrete divergence):

   `(d0'*star1*d0) * alpha = d0'*star1*omega`

   The co-exact and harmonic parts drop out, because `d0'*d1' = 0` and `d0'*star1*h = 0`.
   So Δα = div X: α holds all the sources and sinks.

2. **Stream potential β (dual 0-form, one value per face).** Apply `d1` (the discrete curl):

   `(d1*inv(star1)*d1') * beta = -d1*omega`

   The exact and harmonic parts drop out, because `d1*d0 = 0` and `d1*h = 0`.
   So Δβ = curl X: β holds all the rotation. The sign is chosen so that the co-exact
   field is X_C ≈ n × ∇β (β is the usual stream function). The equivalent primal
   2-form potential is `info.beta2form = inv(star2)*beta`, and
   `info.betaVertex` is β averaged to the vertices for smooth plots.

3. **Harmonic part** h = ω − d0 α + star1⁻¹ d1ᵀ β. Its dimension is
   nE − rank(d0) − rank(d1) (`M.harmonicDimension`): 2g on a closed genus-g surface,
   and (number of boundary loops − 1) on a planar domain.

**Null spaces.** On every connected component, L0 = d0ᵀ star1 d0 has the constants as
its null space. The same holds for L2 = d1 star1⁻¹ d1ᵀ on *closed* components. Both right-hand
sides are automatically compatible, so the system is consistent. The default `'nullspace','pin'`
fixes one value per component and solves exactly. `'nullspace','regularize'` adds a tiny mass term
instead. Either way the potential is then shifted to zero area-weighted mean per component.

**Boundaries.** On components with boundary, L2 is non-singular. The dual vertices
outside the mesh act as β = 0 (Dirichlet), so the co-exact field is tangent to the boundary,
while α has a natural (Neumann) condition. This is the standard "normal / tangential"
choice, and it gives a trivial harmonic space on a disk.

**Diagnostics** in `info`: norms of the three parts, the three normalised pairwise inner
products, residuals of both solves, curl and divergence of h, the divergence (per vertex)
and curl (per face) of ω, and the harmonic dimension. `decomposeVectorField` also returns
the 1-forms (`info.omega`, `omegaE`, `omegaC`, `omegaH`) and `info.gradAlpha`.

## Notes

* The faces must be consistently oriented. The constructor warns about inconsistent
  orientation, non-manifold edges, degenerate faces and isolated vertices.
* On non-Delaunay meshes some cotan weights are negative. The decomposition is still
  exact and orthogonal algebraically, but star1 is no longer positive definite (you get a warning).
  Weights that are exactly zero (for example a right angle at a boundary edge) are clamped to a tiny
  positive value so that star1 can be inverted.

## Test results (`test_DEC`, Octave 8.4)

| case | check | result |
|---|---|---|
| sphere, X = ∇f + n×∇g | reconstruction / orthogonality | ~1e-15 |
| | α error (nSub 2→4) | 2.7e-2 → 1.7e-3 (order 2.0) |
| | β error | 2.9e-2 → 4.3e-3 (order 1.4) |
| disk, f = x²+y/2, g = 1−r² | α error | ~1e-14 (exact) |
| | β error (8→32 rings) | 2.6e-2 → 6.6e-3 (order 1.0, limited by the boundary) |
| torus, d0 f + co-exact + du | dim H = 2, du recovered | ~1e-14 |
| pin vs regularize | same potentials | yes |
