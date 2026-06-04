import { Route, Routes } from "react-router-dom";
import { Sidebar } from "./components/Sidebar";
import { AuthProvider, RequireAuth } from "./auth/AuthContext";
import Login from "./pages/Login";
import PredictionDetail from "./pages/PredictionDetail";
import PredictionsList from "./pages/PredictionsList";
import Analytics from "./pages/Analytics";
import PoliciesSettings from "./pages/PoliciesSettings";
import AddRepository from "./pages/AddRepository";

function ProtectedShell() {
  return (
    <div className="flex h-full min-h-screen bg-slate-50">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<PredictionsList />} />
          <Route path="/predictions/:id" element={<PredictionDetail />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/policies" element={<PoliciesSettings />} />
          <Route path="/repositories" element={<AddRepository />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route
          path="/*"
          element={
            <RequireAuth>
              <ProtectedShell />
            </RequireAuth>
          }
        />
      </Routes>
    </AuthProvider>
  );
}
