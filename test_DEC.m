function test_DEC(doPlot)
%TEST_DEC  Demo and verification of the DEC class and its Hodge decomposition.
%
%   Runs three cases and checks
%     * reconstruction       omega == omegaE + omegaC + omegaH
%     * orthogonality        <.,.>_star1 between the three parts ~ 0
%     * harmonic dimension   0 (sphere, disk), 2 (torus)
%     * potentials           alpha ~ f, beta ~ g for X = grad f + n x grad g
%     * convergence          potential errors shrink under refinement
%
%   test_DEC(true) also plots the fields and potentials.

if nargin < 1, doPlot = false; end
tol = 1e-8;
nFail = 0;
check = @(cond, msg) fprintf('  [%s] %s\n', ternary(cond), msg);

%% ------------------------------------------------------------------
%  1) Unit sphere:  X = grad_S f + n x grad_S g,  f = z + x*y,  g = x - y*z
% -------------------------------------------------------------------
fprintf('\n=== Sphere (closed, genus 0) ===\n');
errA = []; errB = []; hs = [];
for nSub = 2:4
    [V, F] = DEC.icosphere(nSub);
    M = DEC(V, F);
    [X, fT, gT] = sphereField(V);
    [XE, XC, XH, alpha, beta, info] = M.decomposeVectorField(X, 'vertex');

    rec = norm(info.omega - info.omegaE - info.omegaC - info.omegaH) / norm(info.omega);
    orth = max(abs([info.orthoExactCoexact, info.orthoExactHarm, info.orthoCoexactHarm]));
    % potentials vs analytic (both zero-mean)
    fT = fT - sum(diag(M.star0) .* fT) / sum(diag(M.star0));
    gC = sphereStream(M.faceCenter ./ sqrt(sum(M.faceCenter.^2, 2)));
    gC = gC - sum(M.faceArea .* gC) / sum(M.faceArea);
    eA = sqrt(sum(diag(M.star0) .* (alpha - fT).^2) / sum(diag(M.star0) .* fT.^2));
    eB = sqrt(sum(M.faceArea .* (beta - gC).^2) / sum(M.faceArea .* gC.^2));
    errA(end+1) = eA; errB(end+1) = eB; hs(end+1) = mean(sqrt(sum(M.edgeVec.^2, 2)));
    fprintf('  nSub=%d  nV=%5d  h=%.4f  recon=%.1e  orth=%.1e  |H|/|w|=%.1e  err(alpha)=%.2e  err(beta)=%.2e\n', ...
        nSub, M.nV, hs(end), rec, orth, info.normHarmonic / info.normOmega, eA, eB);
end
ok = rec < tol && orth < tol && info.harmonicDimension == 0 && info.normHarmonic / info.normOmega < 1e-6;
nFail = nFail + ~ok; check(ok, 'sphere: reconstruction, orthogonality, no harmonic part');
ok = all(diff(errA) < 0) && all(diff(errB) < 0) && errA(end) < 0.02 && errB(end) < 0.02;
nFail = nFail + ~ok; check(ok, sprintf('sphere: potentials converge (rate alpha %.2f, beta %.2f)', ...
    rate(hs, errA), rate(hs, errB)));
if doPlot
    figure('Name', 'Sphere'); subplot(1,2,1); M.plotScalar(alpha); title('\alpha (divergence potential)');
    subplot(1,2,2); M.plotScalar(beta); title('\beta (stream potential)');
end

%% ------------------------------------------------------------------
%  2) Unit disk:  X = grad f + J grad g,  f = x^2 + y/2,  g = 1 - r^2 (g=0 on boundary)
%     For a domain with boundary this decomposition makes the co-exact part
%     tangent to the boundary (beta = 0 there) and puts no condition on alpha.
% -------------------------------------------------------------------
fprintf('\n=== Planar disk (with boundary) ===\n');
errA = []; errB = []; hs = [];
for nR = [8 16 32]
    [V, F] = DEC.diskMesh(nR);
    M = DEC(V, F);
    x = V(:,1); y = V(:,2);
    gradf = [2*x, 0.5*ones(size(x)), 0*x];
    Jgradg = [2*y, -2*x, 0*x];            % e_z x grad(1 - r^2)
    X = gradf + Jgradg;
    [XE, XC, XH, alpha, beta, info] = M.decomposeVectorField(X, 'vertex');

    rec = norm(info.omega - info.omegaE - info.omegaC - info.omegaH) / norm(info.omega);
    orth = max(abs([info.orthoExactCoexact, info.orthoExactHarm, info.orthoCoexactHarm]));
    w0 = diag(M.star0);
    fT = x.^2 + y/2; fT = fT - sum(w0 .* fT) / sum(w0);
    c = M.faceCenter; gC = 1 - c(:,1).^2 - c(:,2).^2;
    eA = sqrt(sum(w0 .* (alpha - fT).^2) / sum(w0 .* fT.^2));
    eB = sqrt(sum(M.faceArea .* (beta - gC).^2) / sum(M.faceArea .* gC.^2));
    errA(end+1) = eA; errB(end+1) = eB; hs(end+1) = mean(sqrt(sum(M.edgeVec.^2, 2)));
    fprintf('  rings=%2d  nV=%5d  h=%.4f  recon=%.1e  orth=%.1e  |H|/|w|=%.1e  err(alpha)=%.2e  err(beta)=%.2e\n', ...
        nR, M.nV, hs(end), rec, orth, info.normHarmonic / info.normOmega, eA, eB);
end
ok = rec < tol && orth < tol && info.harmonicDimension == 0 && info.normHarmonic / info.normOmega < 1e-6;
nFail = nFail + ~ok; check(ok, 'disk: reconstruction, orthogonality, no harmonic part');
% (alpha is reproduced to machine precision here: f is quadratic and the
%  midpoint flat is exact for linear X, so only require it to be tiny)
ok = max(errA) < 1e-10 && all(diff(errB) < 0) && errB(end) < 0.05;
nFail = nFail + ~ok; check(ok, sprintf('disk: alpha exact, beta converges (rate %.2f)', rate(hs, errB)));
% per-face vector fields: exact part equals grad(alpha) exactly
ok = norm(XE - info.gradAlpha, 'fro') < 1e-10 * norm(XE, 'fro');
nFail = nFail + ~ok; check(ok, 'disk: exact vector field == grad(alpha) per face');
% divergence-free / curl-free checks
ok = norm(M.d1 * info.omegaE) < 1e-10 * norm(info.omega) && ...
     norm(M.d0' * M.star1 * info.omegaC) < 1e-10 * norm(M.d0' * M.star1 * info.omega);
nFail = nFail + ~ok; check(ok, 'disk: exact part curl-free, co-exact part divergence-free');
if doPlot
    figure('Name', 'Disk');
    subplot(2,2,1); M.plotScalar(alpha); view(2); title('\alpha'); hold on; M.plotField(XE); hold off;
    subplot(2,2,2); M.plotScalar(beta);  view(2); title('\beta');  hold on; M.plotField(XC); hold off;
    subplot(2,2,3); M.plotField(XH); view(2); title('harmonic');
    subplot(2,2,4); M.plotScalar(info.divergence); view(2); title('divergence');
end

%% ------------------------------------------------------------------
%  3) Torus: the 1-form du (big-circle angle) is harmonic; a gradient field
%     plus a rotational field plus du must be split back into its three parts.
% -------------------------------------------------------------------
fprintf('\n=== Torus (closed, genus 1) ===\n');
[V, F, U] = DEC.torusMesh(60, 24);
M = DEC(V, F);
du = M.d0 * U; du = mod(du + pi, 2*pi) - pi;              % wrapped angle difference
f = V(:,3).^2 + V(:,1);                                     % arbitrary potentials
g = M.vertexToFace(V(:,2));
omega = M.d0 * f - (M.star1 \ (M.d1' * g)) + du;
[wE, wC, wH, alpha, beta, info] = M.hodgeDecomposition(omega);
rec = norm(omega - wE - wC - wH) / norm(omega);
orth = max(abs([info.orthoExactCoexact, info.orthoExactHarm, info.orthoCoexactHarm]));
fprintf('  nV=%d  dim H=%d  recon=%.1e  orth=%.1e\n', M.nV, info.harmonicDimension, rec, orth);
fprintf('  |wE-d0 f|/|d0 f| = %.2e,  |wH-du|/|du| = %.2e,  |H|/|w| = %.3f\n', ...
    sqrt(M.innerProduct1(wE - M.d0*f, wE - M.d0*f) / M.innerProduct1(M.d0*f, M.d0*f)), ...
    sqrt(M.innerProduct1(wH - du, wH - du) / M.innerProduct1(du, du)), info.normHarmonic / info.normOmega);
ok = rec < tol && orth < tol && info.harmonicDimension == 2 && ...
     M.innerProduct1(wH - du, wH - du) / M.innerProduct1(du, du) < 1e-3;
nFail = nFail + ~ok; check(ok, 'torus: dim H = 2, harmonic part recovered');

%% ------------------------------------------------------------------
%  4) Regularisation instead of pinning gives the same potentials
% -------------------------------------------------------------------
[V, F] = DEC.icosphere(3);
X = sphereField(V);
M1 = DEC(V, F, 'nullspace', 'pin');
M2 = DEC(V, F, 'nullspace', 'regularize');
[~, ~, ~, a1, b1] = M1.decomposeVectorField(X, 'vertex');
[~, ~, ~, a2, b2] = M2.decomposeVectorField(X, 'vertex');
ok = norm(a1 - a2) < 1e-6 * norm(a1) && norm(b1 - b2) < 1e-6 * norm(b1);
nFail = nFail + ~ok; check(ok, 'pin and regularize null-space handling agree');

fprintf('\n%s\n', ternary(nFail == 0, 'ALL TESTS PASSED', sprintf('%d TEST(S) FAILED', nFail)));

end

% ======================================================================
function s = ternary(c, a, b)
if nargin == 1, a = 'PASS'; b = 'FAIL'; end
if c, s = a; else, s = b; end
end

function r = rate(h, e)
p = polyfit(log(h), log(e), 1); r = p(1);
end

function g = sphereStream(P)
g = P(:,1) - P(:,2).*P(:,3);
end

function [X, f, g] = sphereField(V)
% X = grad_S f + n x grad_S g on the unit sphere (n = position)
x = V(:,1); y = V(:,2); z = V(:,3); n = V ./ sqrt(sum(V.^2, 2));
f = z + x.*y;          gf = [y, x, ones(size(z))];
g = x - y.*z;          gg = [ones(size(x)), -z, -y];
proj = @(G) G - sum(G .* n, 2) .* n;
X = proj(gf) + cross(n, proj(gg), 2);
end
