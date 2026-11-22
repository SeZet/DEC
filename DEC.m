classdef DEC < handle
%DEC  Discrete Exterior Calculus on oriented triangle meshes (2-manifolds,
%     closed or with boundary, any genus, possibly several components).
%
%   M = DEC(V, F)            V: nV x 3 (or nV x 2) vertex positions
%                            F: nF x 3 triangle indices (consistently oriented)
%   M = DEC(V, F, 'star0', 'voronoi', 'nullspace', 'pin')
%
%   Discrete forms (column vectors):
%     0-form : nV x 1  (values at vertices)
%     1-form : nE x 1  (integrals along the oriented edges M.E(:,1) -> M.E(:,2))
%     2-form : nF x 1  (integrals over the faces)
%     dual 0-form : nF x 1 (values at circumcentres = dual vertices)
%
%   Operators (sparse):
%     d0    (nE x nV)  exterior derivative on 0-forms (signed incidence)
%     d1    (nF x nE)  exterior derivative on 1-forms (d1*d0 == 0)
%     star0 (nV x nV)  dual cell areas (barycentric or mixed Voronoi)
%     star1 (nE x nE)  cotan weights |e*|/|e| = (cot a + cot b)/2
%     star2 (nF x nF)  1 ./ face area
%
%   Main method:
%     [wE, wC, wH, alpha, beta, info] = M.hodgeDecomposition(omega)
%       omega = wE + wC + wH with
%         wE = d0*alpha                       exact      (curl-free part)
%         wC = -inv(star1)*d1'*beta           co-exact   (divergence-free part)
%         wH = remainder                      harmonic   (d1*wH = 0, d0'*star1*wH = 0)
%       alpha : nV x 1 scalar potential (vertex 0-form)  ->  X_exact   ~ grad(alpha)
%       beta  : nF x 1 stream potential (dual 0-form)    ->  X_coexact ~ n x grad(beta)
%
%   Convenience:
%     [XE, XC, XH, alpha, beta, info] = M.decomposeVectorField(X)
%       X given per vertex (nV x 3) or per face (nF x 3); outputs are per-face
%       vector fields (Whitney reconstruction) plus the same potentials.
%
%   See README.md for the mathematics and conventions.

    properties (SetAccess = private)
        V           % nV x 3 vertex positions
        F           % nF x 3 faces
        E           % nE x 2 edges, oriented from lower to higher vertex index
        d0          % nE x nV
        d1          % nF x nE
        star0       % nV x nV diagonal
        star1       % nE x nE diagonal
        star2       % nF x nF diagonal
        faceArea    % nF x 1
        faceNormal  % nF x 3 unit normals
        faceCenter  % nF x 3 barycentres
        edgeVec     % nE x 3 edge vectors V(E(:,2),:) - V(E(:,1),:)
        isBoundaryEdge   % nE x 1 logical
        isBoundaryVertex % nV x 1 logical
        vertexComponent  % nV x 1 connected component label
        faceComponent    % nF x 1 component label (faces linked by edges)
        closedFaceComponent % logical, per face component: true if it has no boundary
        nullspace   % 'pin' or 'regularize'
        star0Type   % 'barycentric' or 'voronoi'
    end

    properties (Dependent)
        nV
        nE
        nF
    end

    properties (Access = private)
        gradBasis   % nF x 3 x 3 array: gradients of the hat functions per face
    end

    % =====================================================================
    methods
        function obj = DEC(V, F, varargin)
            opts = struct('star0', 'voronoi', 'nullspace', 'pin', 'minweight', 1e-8);
            for k = 1:2:numel(varargin)
                opts.(lower(varargin{k})) = varargin{k+1};
            end
            if size(V,2) == 2, V = [V, zeros(size(V,1),1)]; end
            obj.V = double(V);
            obj.F = double(F);
            obj.nullspace = lower(opts.nullspace);
            obj.star0Type = lower(opts.star0);
            obj.buildTopology();
            obj.buildGeometry(opts.minweight);
        end

        function n = get.nV(obj), n = size(obj.V,1); end
        function n = get.nE(obj), n = size(obj.E,1); end
        function n = get.nF(obj), n = size(obj.F,1); end

        % ---------------------------------------------------------------
        % Laplacians and differential operators
        % ---------------------------------------------------------------
        function L = laplacian0(obj)
            %LAPLACIAN0  Cotan stiffness matrix d0'*star1*d0 (nV x nV, PSD
            %   for Delaunay meshes). Positive semi-definite convention.
            L = obj.d0' * obj.star1 * obj.d0;
        end

        function L = laplacian2dual(obj)
            %LAPLACIAN2DUAL  d1*inv(star1)*d1' acting on dual 0-forms (nF x nF).
            L = obj.d1 * obj.invStar1() * obj.d1';
        end

        function L = laplacian1(obj)
            %LAPLACIAN1  Hodge Laplacian on primal 1-forms (positive convention)
            %   Delta1 = d0*inv(star0)*d0'*star1 + inv(star1)*d1'*star2*d1
            S0i = spdiags(1 ./ diag(obj.star0), 0, obj.nV, obj.nV);
            L = obj.d0 * S0i * obj.d0' * obj.star1 + obj.invStar1() * obj.d1' * obj.star2 * obj.d1;
        end

        function dv = divergence(obj, omega)
            %DIVERGENCE  Pointwise divergence at vertices (nV x 1) of the
            %   vector field represented by the 1-form omega:
            %   div = -inv(star0) * d0' * star1 * omega
            dv = -(obj.d0' * (obj.star1 * omega)) ./ diag(obj.star0);
        end

        function c = curl(obj, omega)
            %CURL  Pointwise (normal) curl per face (nF x 1):  star2 * d1 * omega
            c = obj.star2 * (obj.d1 * omega);
        end

        function ip = innerProduct1(obj, a, b)
            %INNERPRODUCT1  L2 inner product of 1-forms: a' * star1 * b
            ip = a' * (obj.star1 * b);
        end

        function k = harmonicDimension(obj)
            %HARMONICDIMENSION  dim of the discrete harmonic space for this
            %   decomposition: nE - rank(d0) - rank(d1). Equals 2g for a closed
            %   genus-g surface and (#boundary loops - 1) for a planar domain.
            nCompV = max(obj.vertexComponent);
            rank0 = obj.nV - nCompV;
            rank1 = obj.nF - sum(obj.closedFaceComponent);
            k = obj.nE - rank0 - rank1;
        end

        % ---------------------------------------------------------------
        % Hodge-Helmholtz decomposition
        % ---------------------------------------------------------------
        function [omegaE, omegaC, omegaH, alpha, beta, info] = hodgeDecomposition(obj, omega)
            %HODGEDECOMPOSITION  omega = d0*alpha - inv(star1)*d1'*beta + h
            %
            %   [omegaE, omegaC, omegaH, alpha, beta, info] = M.hodgeDecomposition(omega)
            %
            %   omega  : nE x 1 primal 1-form
            %   omegaE : exact part      d0*alpha               (curl-free)
            %   omegaC : co-exact part   -inv(star1)*d1'*beta   (divergence-free)
            %   omegaH : harmonic part   omega - omegaE - omegaC
            %   alpha  : nV x 1 vertex scalar potential, solves
            %              (d0'*star1*d0) alpha = d0'*star1*omega
            %            normalised to zero area-weighted mean per component.
            %   beta   : nF x 1 face (dual 0-form) stream potential, solves
            %              (d1*inv(star1)*d1') beta = -d1*omega
            %            normalised to zero area-weighted mean on closed components.
            %            Equivalent primal 2-form potential: info.beta2form = inv(star2)*beta.
            %   info   : struct with diagnostics (residuals, orthogonality,
            %            divergence/curl, vertex-interpolated beta).
            omega = omega(:);
            assert(numel(omega) == obj.nE, 'omega must have one value per edge (nE x 1).');

            % --- exact part: Poisson problem for alpha on vertices --------
            L0 = obj.laplacian0();
            b0 = obj.d0' * (obj.star1 * omega);
            pin0 = obj.firstOfEachComponent(obj.vertexComponent, true(max(obj.vertexComponent),1));
            alpha = obj.solveWithNullspace(L0, b0, pin0, diag(obj.star0), obj.vertexComponent, ...
                                           true(max(obj.vertexComponent),1));
            omegaE = obj.d0 * alpha;

            % --- co-exact part: Poisson problem for beta on dual vertices --
            % Only closed components have the constant null space; components
            % with boundary are implicitly Dirichlet (beta = 0 outside).
            L2 = obj.laplacian2dual();
            b2 = -(obj.d1 * omega);
            closed = obj.closedFaceComponent;
            pin2 = obj.firstOfEachComponent(obj.faceComponent, closed);
            beta = obj.solveWithNullspace(L2, b2, pin2, obj.faceArea, obj.faceComponent, closed);
            omegaC = -obj.invStar1() * (obj.d1' * beta);

            % --- harmonic remainder ---------------------------------------
            omegaH = omega - omegaE - omegaC;

            if nargout > 5
                nrm = @(w) sqrt(max(obj.innerProduct1(w, w), 0));
                n0 = max(nrm(omega), eps);
                info = struct();
                info.normOmega          = nrm(omega);
                info.normExact          = nrm(omegaE);
                info.normCoexact        = nrm(omegaC);
                info.normHarmonic       = nrm(omegaH);
                info.orthoExactCoexact  = obj.innerProduct1(omegaE, omegaC) / n0^2;
                info.orthoExactHarm     = obj.innerProduct1(omegaE, omegaH) / n0^2;
                info.orthoCoexactHarm   = obj.innerProduct1(omegaC, omegaH) / n0^2;
                info.residualAlpha      = norm(L0*alpha - b0) / max(norm(b0), eps);
                info.residualBeta       = norm(L2*beta - b2) / max(norm(b2), eps);
                info.harmonicCurl       = norm(obj.d1*omegaH) / max(norm(obj.d1*omega), eps);
                info.harmonicDiv        = norm(obj.d0'*obj.star1*omegaH) / max(norm(b0), eps);
                info.harmonicDimension  = obj.harmonicDimension();
                info.divergence         = obj.divergence(omega);   % = div of exact part
                info.curl               = obj.curl(omega);         % = curl of coexact part
                info.beta2form          = beta ./ diag(obj.star2); % primal 2-form version
                info.betaVertex         = obj.faceToVertex(beta);  % for plotting
            end
        end

        function [XE, XC, XH, alpha, beta, info] = decomposeVectorField(obj, X, location)
            %DECOMPOSEVECTORFIELD  Hodge decomposition of a tangent vector field.
            %   [XE, XC, XH, alpha, beta, info] = M.decomposeVectorField(X)
            %   [...] = M.decomposeVectorField(X, 'vertex' | 'face')
            %   X is nV x 3 (per vertex) or nF x 3 (per face). Returns per-face
            %   vector fields (Whitney reconstruction at barycentres) and the
            %   scalar potentials alpha (vertices) and beta (faces).
            %   info additionally contains the 1-forms omega/omegaE/omegaC/omegaH.
            if nargin < 3, location = obj.guessLocation(X); end
            omega = obj.vectorFieldToOneForm(X, location);
            [wE, wC, wH, alpha, beta, info] = obj.hodgeDecomposition(omega);
            XE = obj.oneFormToFaceVectors(wE);
            XC = obj.oneFormToFaceVectors(wC);
            XH = obj.oneFormToFaceVectors(wH);
            info.omega = omega; info.omegaE = wE; info.omegaC = wC; info.omegaH = wH;
            info.gradAlpha = obj.gradient0(alpha);        % == XE exactly
            info.rotGradBeta = obj.rotatedGradientDual(beta); % per-edge estimate
        end

        % ---------------------------------------------------------------
        % Vector fields <-> 1-forms
        % ---------------------------------------------------------------
        function omega = vectorFieldToOneForm(obj, X, location)
            %VECTORFIELDTOONEFORM  Flat operator: integrate X along each edge.
            %   'vertex': X is nV x 3, midpoint rule  omega_e = (X_i+X_j)/2 . e
            %   'face'  : X is nF x 3, average of X_f . e over the adjacent faces
            if nargin < 3, location = obj.guessLocation(X); end
            if size(X,2) == 2, X = [X, zeros(size(X,1),1)]; end
            switch lower(location)
                case 'vertex'
                    Xm = 0.5 * (X(obj.E(:,1),:) + X(obj.E(:,2),:));
                    omega = sum(Xm .* obj.edgeVec, 2);
                case 'face'
                    A = abs(obj.d1)';                     % nE x nF adjacency
                    cnt = full(sum(A, 2));
                    omega = zeros(obj.nE, 1);
                    for c = 1:3
                        omega = omega + (A * X(:,c)) .* obj.edgeVec(:,c);
                    end
                    omega = omega ./ max(cnt, 1);
                otherwise
                    error('location must be ''vertex'' or ''face''.');
            end
        end

        function X = oneFormToFaceVectors(obj, omega)
            %ONEFORMTOFACEVECTORS  Sharp via Whitney interpolation evaluated at
            %   face barycentres (nF x 3). Exact for piecewise-constant fields;
            %   for omega = d0*f it returns grad(f) of the linear interpolant.
            [wij, wjk, wki] = obj.faceEdgeValues(omega);
            G = obj.gradBasis;                     % nF x 3 x 3 (face, vertex, xyz)
            Gi = reshape(G(:,1,:), [], 3); Gj = reshape(G(:,2,:), [], 3); Gk = reshape(G(:,3,:), [], 3);
            X = (wij .* (Gj - Gi) + wjk .* (Gk - Gj) + wki .* (Gi - Gk)) / 3;
        end

        function X = oneFormToVertexVectors(obj, omega)
            %ONEFORMTOVERTEXVECTORS  Area-weighted average of the face vectors
            %   at each vertex (nV x 3), for quiver plots.
            Xf = obj.oneFormToFaceVectors(omega);
            X = obj.faceToVertex(Xf);
        end

        function G = gradient0(obj, f)
            %GRADIENT0  Per-face gradient (nF x 3) of a piecewise linear 0-form.
            G = obj.gradBasis;
            fi = f(obj.F(:,1)); fj = f(obj.F(:,2)); fk = f(obj.F(:,3));
            G = fi .* reshape(G(:,1,:), [], 3) + fj .* reshape(G(:,2,:), [], 3) + fk .* reshape(G(:,3,:), [], 3);
        end

        function R = rotatedGradientDual(obj, beta)
            %ROTATEDGRADIENTDUAL  Edge-based 1-form of n x grad(beta) for a dual
            %   0-form beta (same as the co-exact part): -inv(star1)*d1'*beta.
            R = -obj.invStar1() * (obj.d1' * beta);
        end

        function g = faceToVertex(obj, f)
            %FACETOVERTEX  Area-weighted average of per-face data to vertices.
            W = sparse(obj.F(:), repmat((1:obj.nF)', 3, 1), repmat(obj.faceArea, 3, 1), obj.nV, obj.nF);
            g = (W * f) ./ full(sum(W, 2));
        end

        function f = vertexToFace(obj, g)
            %VERTEXTOFACE  Average of vertex data over each face.
            f = (g(obj.F(:,1),:) + g(obj.F(:,2),:) + g(obj.F(:,3),:)) / 3;
        end

        % ---------------------------------------------------------------
        % Plotting helpers
        % ---------------------------------------------------------------
        function h = plotScalar(obj, s)
            %PLOTSCALAR  Plot a vertex (nV) or face (nF) scalar on the mesh.
            if numel(s) == obj.nV && numel(s) ~= obj.nF
                h = trisurf(obj.F, obj.V(:,1), obj.V(:,2), obj.V(:,3), s, 'EdgeColor', 'none', 'FaceColor', 'interp');
            else
                h = trisurf(obj.F, obj.V(:,1), obj.V(:,2), obj.V(:,3), 'FaceVertexCData', s(:), 'EdgeColor', 'none', 'FaceColor', 'flat');
            end
            axis equal; colorbar;
        end

        function h = plotField(obj, Xf, scale)
            %PLOTFIELD  Quiver plot of a per-face vector field at barycentres.
            if nargin < 3, scale = 1; end
            C = obj.faceCenter;
            h = quiver3(C(:,1), C(:,2), C(:,3), Xf(:,1), Xf(:,2), Xf(:,3), scale);
            axis equal;
        end
    end

    % =====================================================================
    methods (Access = private)
        function buildTopology(obj)
            F = obj.F; nF = size(F,1); nV = size(obj.V,1);
            he = [F(:,[1 2]); F(:,[2 3]); F(:,[3 1])];        % half-edges
            [es, ~] = sort(he, 2);
            [E, ~, ic] = unique(es, 'rows');
            obj.E = E;
            nE = size(E,1);
            sgn = 2 * (he(:,1) == es(:,1)) - 1;                 % +1 if aligned
            faceIdx = repmat((1:nF)', 3, 1);
            obj.d1 = sparse(faceIdx, ic, sgn, nF, nE);
            obj.d0 = sparse([1:nE, 1:nE]', [E(:,1); E(:,2)], [-ones(nE,1); ones(nE,1)], nE, nV);

            % manifold / orientation checks
            edgeFaces = full(sum(abs(obj.d1), 1))';
            if any(edgeFaces > 2)
                warning('DEC:nonmanifold', 'Mesh has %d non-manifold edges.', sum(edgeFaces > 2));
            end
            edgeSum = full(sum(obj.d1, 1))';
            badOrient = (edgeFaces == 2) & (edgeSum ~= 0);
            if any(badOrient)
                warning('DEC:orientation', ['%d interior edges have inconsistent face orientation; ', ...
                    'results will be wrong. Re-orient the faces consistently.'], sum(badOrient));
            end
            obj.isBoundaryEdge = (edgeFaces == 1);
            obj.isBoundaryVertex = false(nV,1);
            obj.isBoundaryVertex(E(obj.isBoundaryEdge, :)) = true;

            % connected components
            Av = abs(obj.d0)' * abs(obj.d0);
            obj.vertexComponent = DEC.components(Av);
            Af = abs(obj.d1) * abs(obj.d1)';
            obj.faceComponent = DEC.components(Af);
            nc = max(obj.faceComponent);
            hasBnd = false(nc,1);
            bf = find(any(abs(obj.d1(:, obj.isBoundaryEdge)), 2));
            hasBnd(unique(obj.faceComponent(bf))) = true;
            obj.closedFaceComponent = ~hasBnd;
            unused = full(sum(abs(obj.d0),1))' == 0;
            if any(unused)
                warning('DEC:isolated', '%d isolated vertices (not in any face).', sum(unused));
            end
        end

        function buildGeometry(obj, minweight)
            V = obj.V; F = obj.F; nF = obj.nF; nE = obj.nE;
            p1 = V(F(:,1),:); p2 = V(F(:,2),:); p3 = V(F(:,3),:);
            N = cross(p2 - p1, p3 - p1, 2);
            dblA = sqrt(sum(N.^2, 2));
            obj.faceArea = dblA / 2;
            if any(dblA <= 0), warning('DEC:degenerate', 'Mesh has degenerate faces.'); end
            obj.faceNormal = N ./ dblA;
            obj.faceCenter = (p1 + p2 + p3) / 3;
            obj.edgeVec = V(obj.E(:,2),:) - V(obj.E(:,1),:);

            % gradients of hat functions: grad(phi_i) = n x e_opp / (2A)
            n = obj.faceNormal;
            G = zeros(nF, 3, 3);
            G(:,1,:) = reshape(cross(n, p3 - p2, 2) ./ dblA, nF, 1, 3);
            G(:,2,:) = reshape(cross(n, p1 - p3, 2) ./ dblA, nF, 1, 3);
            G(:,3,:) = reshape(cross(n, p2 - p1, 2) ./ dblA, nF, 1, 3);
            obj.gradBasis = G;

            % cotangents of the angle at each corner
            cotAt = @(a, b, c) sum((b - a) .* (c - a), 2) ./ dblA;   % angle at a
            c1 = cotAt(p1, p2, p3); c2 = cotAt(p2, p3, p1); c3 = cotAt(p3, p1, p2);

            % star1: edge opposite corner k gets cot/2
            he = [F(:,[2 3]); F(:,[3 1]); F(:,[1 2])];   % edge opposite corner 1,2,3
            [~, idx] = ismember(sort(he, 2), obj.E, 'rows');
            w = accumarray(idx, 0.5 * [c1; c2; c3], [nE, 1]);
            nNeg = sum(w < 0);
            if nNeg > 0
                warning('DEC:nonDelaunay', ['%d edges have negative cotan weight (non-Delaunay mesh). ', ...
                    'The decomposition stays exact/orthogonal algebraically, but star1 is indefinite.'], nNeg);
            end
            tiny = minweight * mean(abs(w));
            small = abs(w) < tiny;
            if any(small)
                w(small) = tiny;     % regularise zero weights (right angles at boundary)
            end
            obj.star1 = spdiags(w, 0, nE, nE);
            obj.star2 = spdiags(1 ./ obj.faceArea, 0, nF, nF);

            % star0: dual vertex areas
            switch obj.star0Type
                case 'barycentric'
                    a0 = accumarray(F(:), repmat(obj.faceArea / 3, 3, 1), [obj.nV, 1]);
                case 'voronoi'
                    a0 = DEC.mixedVoronoiAreas(V, F, obj.faceArea, [c1 c2 c3], obj.nV);
                otherwise
                    error('star0 must be ''barycentric'' or ''voronoi''.');
            end
            a0(a0 <= 0) = eps;
            obj.star0 = spdiags(a0, 0, obj.nV, obj.nV);
        end

        function S = invStar1(obj)
            S = spdiags(1 ./ diag(obj.star1), 0, obj.nE, obj.nE);
        end

        function [wij, wjk, wki] = faceEdgeValues(obj, omega)
            % 1-form values along the oriented face edges i->j, j->k, k->i
            F = obj.F;
            [~, e1] = ismember(sort(F(:,[1 2]), 2), obj.E, 'rows');
            [~, e2] = ismember(sort(F(:,[2 3]), 2), obj.E, 'rows');
            [~, e3] = ismember(sort(F(:,[3 1]), 2), obj.E, 'rows');
            s1 = 2 * (F(:,1) < F(:,2)) - 1;
            s2 = 2 * (F(:,2) < F(:,3)) - 1;
            s3 = 2 * (F(:,3) < F(:,1)) - 1;
            wij = s1 .* omega(e1); wjk = s2 .* omega(e2); wki = s3 .* omega(e3);
        end

        function loc = guessLocation(obj, X)
            if size(X,1) == obj.nV && size(X,1) ~= obj.nF
                loc = 'vertex';
            elseif size(X,1) == obj.nF && size(X,1) ~= obj.nV
                loc = 'face';
            else
                error('Cannot infer whether X is per vertex or per face; pass ''vertex'' or ''face''.');
            end
        end

        function x = solveWithNullspace(obj, L, b, pins, weights, comp, hasNull)
            % Solve the singular-but-consistent system L*x = b. Each component
            % listed in hasNull has a constant null space; it is removed by
            % pinning one value (default) or by a tiny mass regularisation,
            % and the result is shifted to zero weighted mean per component.
            n = size(L, 1);
            switch obj.nullspace
                case 'pin'
                    if ~isempty(pins)
                        keep = true(n,1); keep(pins) = false;
                        x = zeros(n,1);
                        x(keep) = L(keep, keep) \ b(keep);
                    else
                        x = L \ b;
                    end
                case 'regularize'
                    epsReg = 1e-10 * full(max(abs(diag(L)))) / max(weights);
                    m = zeros(n,1);
                    m(ismember(comp, find(hasNull))) = weights(ismember(comp, find(hasNull)));
                    x = (L + epsReg * spdiags(m, 0, n, n)) \ b;
                otherwise
                    error('nullspace must be ''pin'' or ''regularize''.');
            end
            % zero weighted mean on components with a constant null space
            for c = find(hasNull(:))'
                in = (comp == c);
                x(in) = x(in) - sum(weights(in) .* x(in)) / sum(weights(in));
            end
        end
    end

    % =====================================================================
    methods (Static)
        function idx = firstOfEachComponent(comp, mask)
            idx = [];
            for c = find(mask(:))'
                idx(end+1, 1) = find(comp == c, 1); %#ok<AGROW>
            end
        end

        function labels = components(A)
            % connected components of the graph with (symmetric) adjacency A
            n = size(A,1);
            labels = zeros(n,1); c = 0;
            for s = 1:n
                if labels(s) == 0
                    c = c + 1; labels(s) = c; frontier = s;
                    while ~isempty(frontier)
                        nb = find(any(A(:, frontier), 2) & labels == 0);
                        labels(nb) = c;
                        frontier = nb;
                    end
                end
            end
        end

        function a = mixedVoronoiAreas(V, F, area, C, nV)
            % Meyer et al. 2003 mixed Voronoi areas (circumcentric dual for
            % non-obtuse triangles, barycentric-like fallback otherwise).
            p = {V(F(:,1),:), V(F(:,2),:), V(F(:,3),:)};
            L2 = zeros(size(F,1), 3);        % squared length of edge opposite corner k
            L2(:,1) = sum((p{3} - p{2}).^2, 2);
            L2(:,2) = sum((p{1} - p{3}).^2, 2);
            L2(:,3) = sum((p{2} - p{1}).^2, 2);
            obtuse = C < 0;                  % cot < 0  <=> angle > 90 deg
            anyObt = any(obtuse, 2);
            A = zeros(size(F,1), 3);
            for k = 1:3
                k1 = mod(k, 3) + 1; k2 = mod(k + 1, 3) + 1;
                % Voronoi part of corner k: edges k-k1 (opp k2) and k-k2 (opp k1)
                A(:,k) = (L2(:,k2) .* C(:,k2) + L2(:,k1) .* C(:,k1)) / 8;
            end
            for k = 1:3
                A(anyObt, k) = area(anyObt) .* (0.25 + 0.25 * obtuse(anyObt, k));
            end
            a = accumarray(F(:), A(:), [nV, 1]);
        end

        % ------------------------- test meshes ---------------------------
        function [V, F] = icosphere(nSub)
            %ICOSPHERE  Unit sphere by subdividing an icosahedron nSub times.
            t = (1 + sqrt(5)) / 2;
            V = [-1 t 0; 1 t 0; -1 -t 0; 1 -t 0; 0 -1 t; 0 1 t; 0 -1 -t; 0 1 -t; ...
                  t 0 -1; t 0 1; -t 0 -1; -t 0 1];
            F = [1 12 6; 1 6 2; 1 2 8; 1 8 11; 1 11 12; 2 6 10; 6 12 5; 12 11 3; 11 8 7; 8 2 9; ...
                 4 10 5; 4 5 3; 4 3 7; 4 7 9; 4 9 10; 5 10 6; 3 5 12; 7 3 11; 9 7 8; 10 9 2];
            V = V ./ sqrt(sum(V.^2, 2));
            for s = 1:nSub
                nV0 = size(V,1);
                he = [F(:,[1 2]); F(:,[2 3]); F(:,[3 1])];
                [E, ~, ic] = unique(sort(he, 2), 'rows');
                M = (V(E(:,1),:) + V(E(:,2),:)) / 2;
                M = M ./ sqrt(sum(M.^2, 2));
                V = [V; M]; %#ok<AGROW>
                nF0 = size(F,1);
                m = reshape(ic, nF0, 3) + nV0;         % midpoints of edges 12, 23, 31
                F = [F(:,1) m(:,1) m(:,3); m(:,1) F(:,2) m(:,2); m(:,3) m(:,2) F(:,3); m(:,1) m(:,2) m(:,3)];
            end
        end

        function [V, F] = diskMesh(nRings, R)
            %DISKMESH  Delaunay mesh of a disk of radius R with nRings rings.
            if nargin < 2, R = 1; end
            P = [0 0];
            for k = 1:nRings
                n = 6 * k; th = 2*pi*(0:n-1)'/n + (mod(k,2)*pi/n);
                P = [P; (R*k/nRings) * [cos(th) sin(th)]]; %#ok<AGROW>
            end
            F = delaunay(P(:,1), P(:,2));
            % orient counter-clockwise (normal +z)
            a = P(F(:,2),:) - P(F(:,1),:); b = P(F(:,3),:) - P(F(:,1),:);
            neg = (a(:,1).*b(:,2) - a(:,2).*b(:,1)) < 0;
            F(neg, [2 3]) = F(neg, [3 2]);
            V = [P, zeros(size(P,1),1)];
        end

        function [V, F, U] = torusMesh(nU, nV, R, r)
            %TORUSMESH  Torus of revolution with an offset (near-equilateral)
            %   triangulation. nU (even) around the big circle, nV around the tube.
            %   U returns the big-circle angle of each vertex.
            if nargin < 3, R = 1; end
            if nargin < 4, r = 0.4; end
            assert(mod(nU, 2) == 0, 'nU must be even.');
            [J, I] = meshgrid(0:nV-1, 0:nU-1);
            u = 2*pi*I/nU;
            v = 2*pi*(J + 0.5*mod(I,2))/nV;
            X = (R + r*cos(v)).*cos(u); Y = (R + r*cos(v)).*sin(u); Z = r*sin(v);
            V = [X(:) Y(:) Z(:)]; U = u(:);
            id = @(i, j) mod(i, nU) + nU*mod(j, nV) + 1;
            F = zeros(2*nU*nV, 3); t = 0;
            for i = 0:nU-1
                for j = 0:nV-1
                    a = id(i, j); b = id(i, j+1);
                    if mod(i,2) == 0
                        c = id(i+1, j); cp = id(i+1, j-1);
                    else
                        c = id(i+1, j+1); cp = id(i+1, j);
                    end
                    t = t + 1; F(t,:) = [a b c];
                    t = t + 1; F(t,:) = [a c cp];
                end
            end
            % orient outward
            P1 = V(F(:,1),:); n = cross(V(F(:,2),:) - P1, V(F(:,3),:) - P1, 2);
            ctr = (P1 + V(F(:,2),:) + V(F(:,3),:)) / 3;
            ang = atan2(ctr(:,2), ctr(:,1));
            tubeC = R * [cos(ang) sin(ang) zeros(size(ang))];
            neg = sum(n .* (ctr - tubeC), 2) < 0;
            F(neg, [2 3]) = F(neg, [3 2]);
        end
    end
end
