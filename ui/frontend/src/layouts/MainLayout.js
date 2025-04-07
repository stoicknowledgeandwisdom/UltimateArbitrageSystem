import React, { useState } from 'react';
import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import styled from 'styled-components';

// Styled components
const LayoutContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
`;

const Header = styled.header`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
  background-color: #1a1a2e;
  color: white;
  height: 60px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const HeaderLeft = styled.div`
  display: flex;
  align-items: center;
`;

const HeaderRight = styled.div`
  display: flex;
  align-items: center;
  gap: 20px;
`;

const SidebarToggle = styled.button`
  background: none;
  border: none;
  color: white;
  font-size: 1.5rem;
  cursor: pointer;
  margin-right: 15px;
`;

const AppTitle = styled.h1`
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
`;

const UserInfo = styled.div`
  display: flex;
  flex-direction: column;
  align-items: flex-end;
`;

const UserName = styled.span`
  font-weight: 600;
`;

const UserRole = styled.span`
  font-size: 0.8rem;
  opacity: 0.8;
`;

const LogoutButton = styled.button`
  background-color: #e63946;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
  font-weight: 600;
  
  &:hover {
    background-color: #c1121f;
  }
`;

const ContentWrapper = styled.div`
  display: flex;
  flex: 1;
  overflow: hidden;
`;

const Sidebar = styled.aside`
  width: ${props => props.collapsed ? '80px' : '250px'};
  background-color: #16213e;
  color: white;
  transition: width 0.3s ease;
  overflow-y: auto;
`;

const SidebarNav = styled.nav`
  padding: 20px 0;
`;

const NavList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const NavItem = styled.li`
  margin-bottom: 5px;
`;

const StyledNavLink = styled(NavLink)`
  display: flex;
  align-items: center;
  padding: 12px 20px;
  color: #e0e0e0;
  text-decoration: none;
  transition: background-color 0.3s;
  
  &:hover {
    background-color: #233554;
  }
  
  &.active {
    background-color: #0f3460;
    color: white;
    border-left: 4px solid #4cc9f0;
  }
  
  .icon {
    margin-right: ${props => props.collapsed ? '0' : '15px'};
    font-size: 1.2rem;
  }
  
  .label {
    display: ${props => props.collapsed ? 'none' : 'block'};
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
`;

const Main = styled.main`
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background-color: #f8f9fa;
`;

const Footer = styled.footer`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
  height: 40px;
  background-color: #1a1a2e;
  color: white;
  font-size: 0.8rem;
`;

const MainLayout = () => {
  const { currentUser, logout } = useAuth();
  const navigate = useNavigate();
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/login');
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  const toggleSidebar = () => {
    setIsSidebarCollapsed(!isSidebarCollapsed);
  };

  return (
    <LayoutContainer>
      <Header>
        <HeaderLeft>
          <SidebarToggle onClick={toggleSidebar}>
            ☰
          </SidebarToggle>
          <AppTitle>Ultimate Arbitrage System</AppTitle>
        </HeaderLeft>
        <HeaderRight>
          <UserInfo>
            <UserName>{currentUser?.name || 'User'}</UserName>
            <UserRole>{currentUser?.role || 'Trader'}</UserRole>
          </UserInfo>
          <LogoutButton onClick={handleLogout}>
            Logout
          </LogoutButton>
        </HeaderRight>
      </Header>

      <ContentWrapper>
        <Sidebar collapsed={isSidebarCollapsed}>
          <SidebarNav>
            <NavList>
              <NavItem>
                <StyledNavLink to="/" end collapsed={isSidebarCollapsed}>
                  <span className="icon">📊</span>
                  <span className="label">Dashboard</span>
                </StyledNavLink>
              </NavItem>
              <NavItem>
                <StyledNavLink to="/trading" collapsed={isSidebarCollapsed}>
                  <span className="icon">📈</span>
                  <span className="label">Trading View</span>
                </StyledNavLink>
              </NavItem>
              <NavItem>
                <StyledNavLink to="/strategies" collapsed={isSidebarCollapsed}>
                  <span className="icon">⚙️</span>
                  <span className="label">Strategy Management</span>
                </StyledNavLink>
              </NavItem>
              <NavItem>
                <StyledNavLink to="/wallet" collapsed={isSidebarCollapsed}>
                  <span className="icon">💰</span>
                  <span className="label">Wallet Management</span>
                </StyledNavLink>
              </NavItem>
              <NavItem>
                <StyledNavLink to="/monitoring" collapsed={isSidebarCollapsed}>
                  <span className="icon">🔍</span>
                  <span className="label">System Monitoring</span>
                </StyledNavLink>
              </NavItem>
            </NavList>
          </SidebarNav>
        </Sidebar>

        <Main>
          <Outlet />
        </Main>
      </ContentWrapper>

      <Footer>
        <div>Ultimate Arbitrage System &copy; {new Date().getFullYear()}</div>
        <div>Version 1.0.0</div>
      </Footer>
    </LayoutContainer>
  );
};

export default MainLayout;

