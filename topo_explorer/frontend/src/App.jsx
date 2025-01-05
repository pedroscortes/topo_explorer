import React from 'react';
import ManifoldExplorer from '../components/ManifoldExplorer';

function App() {
  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">
            Topological Space Explorer
          </h1>
          <p className="mt-2 text-gray-600">
            Interactive visualization of geometric learning on manifolds
          </p>
        </header>

        <main>
          <ManifoldExplorer />
        </main>
      </div>
    </div>
  );
}

export default App;