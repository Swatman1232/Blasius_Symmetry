function Tjet = append_blasius_exact_jets(T)
%APPEND_BLASIUS_EXACT_JETS Add analytic Blasius jet columns for QA only.

    x = T.x(:);
    y = T.y(:);
    Ue = T.Ue(1);
    nu = T.nu(1);

    eta = y.*sqrt(Ue./(2*nu*x));
    etaMax = max(10,1.05*max(eta));
    [etaSol,F,Fp,Fpp,~] = solve_blasius_ode45_mod(etaMax);

    F0 = interp1(etaSol,F,eta,'pchip','extrap');
    F1 = interp1(etaSol,Fp,eta,'pchip','extrap');
    F2 = interp1(etaSol,Fpp,eta,'pchip','extrap');
    F3 = -F0.*F2;

    A = sqrt(Ue/(2*nu));
    B = sqrt(2*nu*Ue);

    Tjet = T;
    Tjet.psi = B.*sqrt(x).*F0;
    Tjet.psi_x = B./(2*sqrt(x)).*(F0 - eta.*F1);
    Tjet.psi_y = Ue.*F1;

    Tjet.psi_xx = B./(4*x.^(3/2)).*(-F0 + eta.*F1 + eta.^2.*F2);
    Tjet.psi_xy = -Ue.*eta.*F2./(2*x);
    Tjet.psi_yy = Ue.*A.*F2./sqrt(x);

    Tjet.psi_xxx = B./(8*x.^(5/2)).*(3*F0 - 3*eta.*F1 ...
                    - 6*eta.^2.*F2 - eta.^3.*F3);
    Tjet.psi_xxy = Ue./(4*x.^2).*(3*eta.*F2 + eta.^2.*F3);
    Tjet.psi_xyy = -Ue.*A./(2*x.^(3/2)).*(F2 + eta.*F3);
    Tjet.psi_yyy = Ue.*A.^2.*F3./x;
end